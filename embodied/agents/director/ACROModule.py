import sys
import numpy as np
import embodied
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import nets
from . import tfutils

class ACROModule(tfutils.Module):
    def __init__(self, act_space, obs_space, config):
        super().__init__()
        self.config = config
        
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if k.startswith('image')}

        # ACRO Model dimensions
        size_H, size_W = self.config.env.size
        channels = 1 if self.config.env.gray else 3
        self.acro_state_size = (size_H, size_W, channels * self.config.frame_stack)

        # Encoder and heads
        self.acro_encoder_backbone = nets.MultiEncoder(
            {"frame_stack": self.acro_state_size},
            **self.config.acro_encoder_backbone
        )
        self.acro_embedding_head = nets.Dense(
            units=self.config.acro_embed_size,
            **self.config.acro_embedding_head
        )
        self.acro_action_head = nets.MLP(
            act_space['action'].shape,
            **self.config.acro_action_head
        )
        
        self.decoder_backbone = nets.MultiDecoder(
            shapes,
            **self.config.acro_decoder_backbone
        )

        # Translation layer
        # self.wm_to_acro_backbone = nets.MLP(
        #     shape=(self.config.acro_embed_size,),
        #     **self.config.wm_to_acro_backbone
        # )
        # self.opt_wm = tfutils.Optimizer('acro_wm', **self.config.acro_opt)
        self.opt_act = tfutils.Optimizer('acro_action', **self.config.acro_opt)
        self.opt_decoder = tfutils.Optimizer('acro_decoder', **self.config.acro_opt)

    @tf.function
    def embed_acro(self, frame_stack):
        hidden = self.acro_encoder_backbone({"frame_stack": frame_stack})
        return self.acro_embedding_head(hidden)

    # def translate_wm(self, wm_state):
    #     return self.wm_to_acro_backbone(wm_state)

    # Images: [T, H, W, C], terminal: [T,], actions: [T, A]
    def extract_batch_from_traj(self, images, is_terminal, actions):
        assert len(images.shape) == 4, "Images should be of shape [T, H, W, C]"
        assert len(is_terminal.shape) == 1, "is_terminal should be of shape [T,]"
        assert len(actions.shape) == 2, "Actions should be of shape [T, A]"
        assert images.shape[0] == is_terminal.shape[0] == actions.shape[0], "All inputs must have the same time dimension T"

        frame_k = tf.constant(self.config.frame_stack, dtype=tf.int32)

        # 1) Stack frames: [T-k+1, k, H, W, C]
        stacked_frames = tf.signal.frame(
            images,
            frame_length=frame_k,
            frame_step=1,
            axis=0
        )

        term_windows = tf.signal.frame(
            is_terminal,
            frame_length=frame_k,
            frame_step=1,
            axis=0
        )

        term_masks = tf.cast(tf.math.cumprod(~tf.cast(term_windows, tf.int32), axis=1, reverse=True), tf.bool)

        masked_frames = stacked_frames * tf.cast(term_masks, stacked_frames.dtype)[..., None, None, None]

        masked_frames = tf.transpose(masked_frames, perm=[0, 2, 3, 1, 4])
        shape = tf.shape(masked_frames)
        t_, h, w, k, c = shape[0], shape[1], shape[2], shape[3], shape[4]
        stacked_frames = tf.reshape(masked_frames, [t_, h, w, k * c])

        state_t = []
        state_tk = []
        acts = []

        k = tf.constant(self.config.acro_k_step, dtype=tf.int32)
        for t in range(t_ - k + 1):
            if tf.reduce_any(tf.cast(term_windows[t], tf.bool)):
                continue
            # Stack frames for t and t + k
            cur = stacked_frames[t]
            cur = tf.ensure_shape(cur, [h, w, c * frame_k])

            fut = stacked_frames[t + k]
            fut = tf.ensure_shape(fut, [h, w, c * frame_k])

            state_t.append(cur)
            state_tk.append(fut)
            acts.append(actions[t + k + frame_k - 1])

        return tf.stack(state_t), tf.stack(state_tk), tf.stack(acts)

    def get_acro_dataset(self, images, is_terminal, actions):
        batch_t, batch_tk, batch_actions = [], [], []
        for i in range(images.shape[1]):
            # Extract batch for each trajectory
            batch_images = images[:, i, ...]
            batch_is_terminal = is_terminal[:, i, ...]
            batch_action = actions[:, i, ...]

            states_t, states_tk, actions_t = self.extract_batch_from_traj(batch_images, batch_is_terminal, batch_action)
            batch_t.append(states_t)
            batch_tk.append(states_tk)
            batch_actions.append(actions_t)

        # Stack all batches
        batch_t = tf.concat(batch_t, axis=0)  # [B, T, H, W, C * k]
        batch_tk = tf.concat(batch_tk, axis=0)  # [B, T, H, W, C * k]
        batch_actions = tf.concat(batch_actions, axis=0)  # [B, T, A]
        return batch_t, batch_tk, batch_actions

    @tf.function
    def _stack_frames(self, images, is_terminal, t):
        frame_k = tf.constant(self.config.frame_stack, dtype=tf.int32)
        start = t - frame_k + 1
        window = images[start : t + 1]  # [k, B, H, W, C]
        # If there is a terminal state that is not the last frame, mask it out to zeros
        if tf.reduce_any(is_terminal[start : t + 1] == 1):
            last_terminal = tf.where(is_terminal[start : t])[-1, 0]
            reset = tf.zeros_like(window[0])
            window = tf.concat([
                tf.repeat(tf.expand_dims(reset, 0), last_terminal + 1, axis=0),
                window[last_terminal + 1:]
            ], axis=0)
        permuted = tf.transpose(window, perm=[1, 2, 3, 4, 0])  # [B, H, W, C, k]
        shape = tf.shape(permuted)
        B, H, W, C, K = shape[0], shape[1], shape[2], shape[3], shape[4]
        return tf.reshape(permuted, [B, H, W, C * K])

    @tf.function
    def get_training_data(self, images, is_terminal, actions):
        """Correctly handle resets and maintain two sets of states for t and t + k."""
        # returns a list of states for t and t + k
        k = tf.constant(self.config.acro_k_step, dtype=tf.int32)
        T = tf.shape(images)[0]
        B = tf.shape(images)[1]
        H = tf.shape(images)[2]
        W = tf.shape(images)[3]
        C = tf.shape(images)[4]

        # Do NOT reshape images, is_terminal, actions
        num_steps = T - k
        states_t_ta = tf.TensorArray(tf.float32, size=num_steps)
        states_tk_ta = tf.TensorArray(tf.float32, size=num_steps)
        actions_t_ta = tf.TensorArray(tf.float32, size=num_steps)

        t = tf.constant(0, dtype=tf.int32)
        def cond(t, *_):
            return t < num_steps

        def body(t, states_t_ta, states_tk_ta, actions_t_ta):
            cur = self._stack_frames(images, is_terminal, t)
            cur = tf.ensure_shape(cur, [B, H, W, C * self.config.frame_stack])
            acro_cur = self.embed_acro(cur)

            fut = self._stack_frames(images, is_terminal, t + k)
            fut = tf.ensure_shape(fut, [B, H, W, C * self.config.frame_stack])
            acro_fut = self.embed_acro(fut)

            states_t_ta = states_t_ta.write(t, acro_cur[0])
            states_tk_ta = states_tk_ta.write(t, acro_fut[0])
            actions_t_ta = actions_t_ta.write(t, actions[t])
            return t + 1, states_t_ta, states_tk_ta, actions_t_ta

        _, states_t_ta, states_tk_ta, actions_t_ta = tf.while_loop(
            cond,
            body,
            [t, states_t_ta, states_tk_ta, actions_t_ta]
        )

        states_t = states_t_ta.stack()
        states_tk = states_tk_ta.stack()
        actions_t = actions_t_ta.stack()
        return states_t, states_tk, actions_t
            

    
    @tf.function
    def train(self, data):
        # Unpack
        images = data['image']  # [T, B, H, W, C]
        actions = data['action']
        is_terminal = data['is_terminal']  # [T, B, 1]

        # TensorArrays for metrics
        # ta_wm = tf.TensorArray(tf.float32, size=N)
        ta_action = tf.TensorArray(tf.float32, size=1)
        ta_decoder = tf.TensorArray(tf.float32, size=1)
        idx = tf.constant(0, tf.int32)

        breakpoint()
        # Action update
        with tf.GradientTape() as action_tape:
            states_t, states_tk, actions_t = self.get_acro_dataset(images, is_terminal, actions)
            embed_t = self.embed_acro(states_t)
            embed_tk = self.embed_acro(states_tk)
            action_dist = self.acro_action_head({
                'state_t': embed_t,
                'state_tk': embed_tk
            })
            action_loss = -tf.reduce_mean(
                action_dist.log_prob(actions_t)
            )
            
        self.opt_act(
            action_tape,
            action_loss,
            [
                self.acro_action_head,
                self.acro_embedding_head,
                self.acro_encoder_backbone
            ]
        )
        
        # with tf.GradientTape() as decoder_tape:
        #     # Reconstruct the future frame
        #     T, B, H, W, C = tf.shape(images)
        #     images = tf.reshape(images, [T * B, H, W, C])  # [T * B, H, W, C]
        #
        #     reconstructed = self.decoder_backbone({"acro": self.embed_acro(images)})
        #
        #     # Calculate reconstruction loss
        #     reconstruction_loss = -tf.reduce_mean(
        #         reconstructed["image"].log_prob(tf.cast(images, reconstructed['image'].dtype))
        #     )
        
        # self.opt_decoder(
        #     decoder_tape,
        #     reconstruction_loss,
        #     [self.decoder_backbone]
        # )

        # Record losses
        # ta_wm = ta_wm.write(idx, wm_loss)
        ta_action = ta_action.write(idx, action_loss)
        # ta_decoder = ta_decoder.write(idx, reconstruction_loss)
        idx += 1

        # Stack and return
        # wm_losses = ta_wm.stack()
        action_losses = ta_action.stack()
        decoder_losses = ta_decoder.stack()
        return {'acro_action_loss': action_losses, "acro_decoder_loss": decoder_losses}
