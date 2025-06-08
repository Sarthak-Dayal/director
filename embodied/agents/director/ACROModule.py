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

        self.opt_act = tfutils.Optimizer('acro_action', **self.config.acro_opt)
        self.opt_decoder = tfutils.Optimizer('acro_decoder', **self.config.acro_opt)


    def _trim(self, x, n):
        "Remove the padding again."
        return x[:n]
    
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

        frame_k = self.config.frame_stack

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
        shape = masked_frames.shape
        t_, h, w, k, c = shape[0], shape[1], shape[2], shape[3], shape[4]
        stacked_frames = tf.reshape(masked_frames, [t_, h, w, k * c])

        state_t = tf.TensorArray(stacked_frames.dtype, size=0, dynamic_size=True)
        state_tk = tf.TensorArray(stacked_frames.dtype, size=0, dynamic_size=True)
        acts = tf.TensorArray(actions.dtype, size=0, dynamic_size=True)

        k = tf.constant(self.config.acro_k_step, dtype=tf.int32)
        for t in tf.range(t_ - k):
            if tf.reduce_any(tf.cast(is_terminal[t + frame_k - 1 : t + frame_k + k - 1], tf.bool)):
                # If no valid frames were found, write a zero frame
                # Removes a warning about conditionally empty TensorArrays and also does padding.
                state_t = state_t.write(0, tf.zeros_like(stacked_frames[0], dtype=stacked_frames.dtype))
                state_tk = state_tk.write(0, tf.zeros_like(stacked_frames[0], dtype=stacked_frames.dtype))
                acts = acts.write(0, tf.zeros_like(actions[0], dtype=acts.dtype))
                continue
            
            # Stack frames for t and t + k
            cur = stacked_frames[t]
            cur = tf.ensure_shape(cur, [h, w, c * frame_k])

            fut = stacked_frames[t + k]
            fut = tf.ensure_shape(fut, [h, w, c * frame_k])
            
            act = actions[t + frame_k - 1]

            state_t = state_t.write(t, cur)
            state_tk = state_tk.write(t, fut)
            acts = acts.write(t, act)

        return state_t.stack(), state_tk.stack(), acts.stack()
    
    def get_acro_dataset(self, images, is_terminal, actions):
        batch_t, batch_tk, batch_actions = tf.TensorArray(images.dtype, size=0, dynamic_size=True), tf.TensorArray(images.dtype, size=0, dynamic_size=True), tf.TensorArray(actions.dtype, size=0, dynamic_size=True)
        for i in range(images.shape[1]):
            # Extract batch for each trajectory
            batch_images = images[:, i, ...]
            batch_is_terminal = is_terminal[:, i, ...]
            batch_action = actions[:, i, ...]

            states_t, states_tk, actions_t = self.extract_batch_from_traj(batch_images, batch_is_terminal, batch_action)
            batch_t = batch_t.write(i, states_t)
            batch_tk = batch_tk.write(i, states_tk)
            batch_actions = batch_actions.write(i, actions_t)


        # Stack all batches
        batch_t_stacked = batch_t.stack()
        batch_tk_stacked = batch_tk.stack()
        action_t_stacked = batch_actions.stack()
        
        shape_batch = tf.shape(batch_t_stacked)
        shape_action = tf.shape(action_t_stacked)
        
        new_shape = tf.concat([[shape_batch[0] * shape_batch[1]], shape_batch[2:]], axis=0)
        batch_t_reshaped = tf.reshape(batch_t_stacked, new_shape)
        batch_tk_reshaped = tf.reshape(batch_tk_stacked, new_shape)
        action_t_reshaped = tf.reshape(action_t_stacked, tf.concat([[shape_batch[0] * shape_batch[1]], shape_action[2:]], axis=0))
        
        return batch_t_reshaped, batch_tk_reshaped, action_t_reshaped

    def train(self, data):
        images = data['image']  # [T, B, H, W, C]
        actions = data['action']
        is_terminal = data['is_terminal']  # [T, B, 1]

        # TensorArrays for metrics
        # ta_wm = tf.TensorArray(tf.float32, size=N)
        ta_action = tf.TensorArray(actions.dtype, size=1)
        ta_decoder = tf.TensorArray(tf.float32, size=1)
        idx = tf.constant(0, tf.int32)

        
        # Action update
        with tf.GradientTape() as action_tape:
            states_t, states_tk, actions_t = self.get_acro_dataset(images, is_terminal, actions)
            MAX_B = data['image'].shape[0] * data['image'].shape[1]  # T * B
            num_valid_windows = data['image'].shape[0] - self.config.frame_stack + 1
            num_valid_states_per_batch = num_valid_windows - self.config.acro_k_step
            states_t_shape = data['image'].shape[1] * num_valid_states_per_batch
            
            states_t = tf.ensure_shape(states_t, [states_t_shape, self.acro_state_size[0], self.acro_state_size[1], self.acro_state_size[2]])
            states_tk = tf.ensure_shape(states_tk, [states_t_shape, self.acro_state_size[0], self.acro_state_size[1], self.acro_state_size[2]])
            actions_t = tf.ensure_shape(actions_t, [states_t_shape, actions.shape[2]])
            
            embed_t  = self.embed_acro(states_t)
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
        
        with tf.GradientTape() as decoder_tape:
            # Reconstruct the future frame
            T, B, H, W, C = tf.shape(images)
            images = tf.reshape(images, [T * B, H, W, C])  # [T * B, H, W, C]
        
            reconstructed = self.decoder_backbone({"acro": self.embed_acro(images)})
        
            # Calculate reconstruction loss
            reconstruction_loss = -tf.reduce_mean(
                reconstructed["image"].log_prob(tf.cast(images, reconstructed['image'].dtype))
            )
        
        self.opt_decoder(
            decoder_tape,
            reconstruction_loss,
            [self.decoder_backbone]
        )

        # Record losses
        # ta_wm = ta_wm.write(idx, wm_loss)
        ta_action = ta_action.write(idx, action_loss)
        # ta_decoder = ta_decoder.write(idx, reconstruction_loss)
        idx += 1

        # Stack and return
        # wm_losses = ta_wm.stack()
        action_losses = ta_action.stack()
        decoder_losses = ta_decoder.stack()
        return {'acro_action_loss': action_losses, 'acro_decoder_loss': decoder_losses}
