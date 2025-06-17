import sys
import unittest
import pathlib

directory = pathlib.Path(__file__)
try:
    import google3  # noqa
except ImportError:
    directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import numpy as np
import embodied
import ruamel.yaml as yaml
import tensorflow as tf
import random
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

        new_action_head = self.config.acro_action_head.update({"dist": "onehot" if act_space['action'].discrete else "mse"})
        self.config = self.config.update({"acro_action_head": new_action_head})

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
        mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)

        k = tf.constant(self.config.acro_k_step, dtype=tf.int32)
        for t in tf.range(t_ - k):
            if tf.reduce_any(tf.cast(is_terminal[t + frame_k - 1 : t + frame_k + k - 1], tf.bool)):
                # If no valid frames were found, write a zero frame
                # Removes a warning about conditionally empty TensorArrays and also does padding.
                state_t = state_t.write(t, tf.zeros_like(stacked_frames[0], dtype=stacked_frames.dtype))
                state_tk = state_tk.write(t, tf.zeros_like(stacked_frames[0], dtype=stacked_frames.dtype))
                acts = acts.write(t, tf.zeros_like(actions[0], dtype=acts.dtype))
                mask = mask.write(t, tf.constant(False, dtype=tf.bool))
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
            mask = mask.write(t, tf.constant(True, dtype=tf.bool))

        return state_t.stack(), state_tk.stack(), acts.stack(), mask.stack()
    
    def get_acro_dataset(self, images, is_terminal, actions):
        batch_t, batch_tk, batch_actions, batch_mask = tf.TensorArray(images.dtype, size=0, dynamic_size=True), tf.TensorArray(images.dtype, size=0, dynamic_size=True), tf.TensorArray(actions.dtype, size=0, dynamic_size=True), tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        for i in range(images.shape[1]):
            # Extract batch for each trajectory
            batch_images = images[:, i, ...]
            batch_is_terminal = is_terminal[:, i, ...]
            batch_action = actions[:, i, ...]

            states_t, states_tk, actions_t, mask_t = self.extract_batch_from_traj(batch_images, batch_is_terminal, batch_action)
            batch_t = batch_t.write(i, states_t)
            batch_tk = batch_tk.write(i, states_tk)
            batch_actions = batch_actions.write(i, actions_t)
            batch_mask = batch_mask.write(i, mask_t)


        # Stack all batches
        batch_t_stacked = batch_t.stack()
        batch_tk_stacked = batch_tk.stack()
        action_t_stacked = batch_actions.stack()
        mask_t_stacked = batch_mask.stack()
        
        shape_batch = tf.shape(batch_t_stacked)
        shape_action = tf.shape(action_t_stacked)
        
        new_shape = tf.concat([[shape_batch[0] * shape_batch[1]], shape_batch[2:]], axis=0)
        batch_t_reshaped = tf.reshape(batch_t_stacked, new_shape)
        batch_tk_reshaped = tf.reshape(batch_tk_stacked, new_shape)
        action_t_reshaped = tf.reshape(action_t_stacked, tf.concat([[shape_batch[0] * shape_batch[1]], shape_action[2:]], axis=0))
        mask_t_reshaped = tf.reshape(mask_t_stacked, tf.concat([[shape_batch[0] * shape_batch[1]], mask_t_stacked.shape[2:]], axis=0))
        
        return batch_t_reshaped, batch_tk_reshaped, action_t_reshaped, mask_t_reshaped

    def train(self, data):
        images = data['image']  # [B, T, H, W, C]
        actions = data['action'] # [B, T, A]
        is_terminal = data['is_terminal']  # [B, T]

        # Reshape everything to [T, B, ...]
        images = tf.transpose(images, perm=[1, 0, 2, 3, 4])  # [T, B, H, W, C]
        actions = tf.transpose(actions, perm=[1, 0, 2])  # [T, B, A]
        is_terminal = tf.transpose(is_terminal, perm=[1, 0])  # [T, B]

        # TensorArrays for metrics
        # ta_wm = tf.TensorArray(tf.float32, size=N)
        ta_action = tf.TensorArray(actions.dtype, size=1)
        ta_decoder = tf.TensorArray(tf.float32, size=1)
        idx = tf.constant(0, tf.int32)

        
        # Action update
        with tf.GradientTape() as action_tape:
            states_t, states_tk, actions_t, mask_t = self.get_acro_dataset(images, is_terminal, actions)
            states_tk = states_tk * 0
            num_valid_windows = data['image'].shape[1] - self.config.frame_stack + 1
            num_valid_states_per_batch = num_valid_windows - self.config.acro_k_step
            states_t_shape = data['image'].shape[0] * num_valid_states_per_batch
            
            states_t = tf.ensure_shape(states_t, [states_t_shape, self.acro_state_size[0], self.acro_state_size[1], self.acro_state_size[2]])
            states_tk = tf.ensure_shape(states_tk, [states_t_shape, self.acro_state_size[0], self.acro_state_size[1], self.acro_state_size[2]])
            actions_t = tf.ensure_shape(actions_t, [states_t_shape, actions.shape[2]])
            
            embed_t  = self.embed_acro(states_t)
            embed_tk = self.embed_acro(states_tk)
            
            action_dist = self.acro_action_head({
                'state_t': embed_t,
                'state_tk': embed_tk
            })

            valid = tf.cast(mask_t, tf.float32)
            action_loss = -tf.reduce_mean(
                action_dist.log_prob(actions_t) * valid / tf.reduce_sum(valid)
            )

        action_pred = tf.cast(
                tf.equal(
                    tf.argmax(action_dist.mode(), axis=-1),
                    tf.argmax(actions_t, axis=-1)
                ),
                tf.float32
            ) * tf.cast(mask_t, tf.float32)

        action_pred_acc = tf.reduce_sum(action_pred) / tf.reduce_sum(tf.cast(mask_t, tf.float32))

        self.opt_act(
            action_tape,
            action_loss,
            [
                self.acro_action_head,
                self.acro_embedding_head,
                self.acro_encoder_backbone
            ]
        )

        embed_t = self.embed_acro(states_t)
        with tf.GradientTape() as decoder_tape:
            # Reconstruct the future frame
            T, B, H, W, C = images.shape
            images = tf.reshape(images, [T * B, H, W, C])  # [T * B, H, W, C]
        
            reconstructed = self.decoder_backbone({"acro": embed_t})
        
            # Calculate reconstruction loss
            reconstruction_loss = -tf.reduce_mean(
                reconstructed["image"].log_prob(tf.cast(images[self.config.frame_stack - 1: self.config.frame_stack + states_t_shape - 1], reconstructed['image'].dtype))
            )
        
        self.opt_decoder(
            decoder_tape,
            reconstruction_loss,
            [self.decoder_backbone]
        )

        # Record losses
        # ta_wm = ta_wm.write(idx, wm_loss)
        ta_action = ta_action.write(idx, action_loss)
        ta_decoder = ta_decoder.write(idx, reconstruction_loss)
        idx += 1

        # Stack and return
        # wm_losses = ta_wm.stack()
        action_losses = ta_action.stack()
        decoder_losses = ta_decoder.stack()

        metrics = {'acro_action_loss': action_losses, 'acro_pred_acc': action_pred_acc, 'acro_decoder_loss': decoder_losses}

        return metrics


    def report(self, data):
        images = data['image']  # [T, B, H, W, C]
        actions = data['action']
        is_terminal = data['is_terminal']  # [T, B, 1]

        states_t, _, _, _ = self.get_acro_dataset(images, is_terminal, actions)
        num_valid_windows = data['image'].shape[1] - self.config.frame_stack + 1
        num_valid_states_per_batch = num_valid_windows - self.config.acro_k_step
        states_t_shape = data['image'].shape[0] * num_valid_states_per_batch

        states_t = tf.ensure_shape(states_t, [states_t_shape, self.acro_state_size[0], self.acro_state_size[1],
                                              self.acro_state_size[2]])

        T, B, H, W, C = images.shape
        images = tf.reshape(images, [T * B, H, W, C])  # [T * B, H, W, C]

        reconstructed = self.decoder_backbone({"acro": self.embed_acro(states_t)})

        random_img_idx = random.randint(0, states_t_shape - 1)

        img = images[self.config.frame_stack - 1 + random_img_idx]
        img = tf.expand_dims(img, 0)

        recon = reconstructed['image'].mean()[random_img_idx]
        recon = tf.expand_dims(recon, 0)

        metrics = {f'acro_recon_image': tf.concat([tf.cast(img, recon.dtype), recon], 1)}
        return metrics

class ACROModuleTest(unittest.TestCase):

    def test_dataset_gen(self):
        from . import agent as agnt

        def gen_images_actions(T, B):
            # 1) Create 1-D indices [0,1,2,…,T*B-1]
            idx = tf.range(T * B, dtype=tf.float32)

            # 2) Build `images` of shape [T*B,H,W,C] where each slice = its index
            images = tf.reshape(
                tf.transpose(
                    tf.reshape(idx, [B, T]), perm=[1, 0]
                ), [T, B, 1, 1, 1]
            )
            images = tf.broadcast_to(images, [T, B, 64, 64, 3])

            actions = tf.reshape(
                tf.transpose(
                    tf.reshape(idx, [B, T]), perm=[1, 0]
                ), [T, B, 1]
            )

            return images, actions

        def gen_terminals_zeros(T, B):
            # Generate a tensor of shape [T, B] filled with zeros
            return tf.zeros((T, B), dtype=tf.bool)

        def gen_terminals_last(T, B):
            # all False for first T-1 steps, all True at the last step
            zeros = tf.zeros((T - 1, B), dtype=tf.bool)
            ones = tf.concat([tf.zeros((1, B - 1), dtype=tf.bool), tf.ones((1, 1), dtype=tf.bool)], axis=1)
            return tf.concat([zeros, ones], axis=0)

        def gen_terminals_random(T, B):
            # Generate a tensor of shape [T, B] with random boolean values
            return tf.cast(tf.random.uniform((T, B), maxval=2, dtype=tf.int32), tf.bool)

        def mini_test(T, B, frame_stack, k_step, terminal_gen=gen_terminals_last):
            print("-" * 50)
            print(f"Testing with T={T}, B={B}, frame_stack={frame_stack}, k_step={k_step}")
            print("-" * 50)

            config = embodied.Config(agnt.Agent.configs['defaults'])
            config = config.update({"frame_stack": frame_stack, "acro_k_step": k_step})

            acro_module = ACROModule(
                act_space={'action': embodied.Space(np.uint8)},
                obs_space={'image': embodied.Space(np.uint8, (64, 64, 3))},
                config=config
            )

            images, actions = gen_images_actions(T, B)
            is_terminal = terminal_gen(T, B)

            # print("Images: ", images[:, :, 0, 0, 0])
            # print("Actions: ", actions[:, :, 0])
            # print("Is Terminal: ", is_terminal)

            batch_t, batch_tk, action_t, mask_t = acro_module.get_acro_dataset(images, is_terminal, actions)

            print("Batch T:", batch_t[:, 0, 0, -1])
            print("Batch Tk:", batch_tk[:, 0, 0, -1])
            print("Action T:", action_t[:, 0])

            # Convert to NumPy
            batch_last = batch_t[:, 0, 0, -1].numpy()
            action_vals = action_t[:, 0].numpy()
            batch_frame = batch_t[:, 0, 0, :].numpy()
            tk_offset = (batch_tk[:, 0, 0, :] - config.acro_k_step).numpy()
            mask = mask_t.numpy().reshape(-1)

            # Filter only valid entries where mask is True
            batch_last_valid = batch_last[mask]
            action_valid = action_vals[mask]
            batch_frame_valid = batch_frame[mask]
            tk_offset_valid = tk_offset[mask]

            # Assertions on filtered data
            self.assertListEqual(batch_last_valid.tolist(), action_valid.tolist())
            self.assertListEqual(batch_frame_valid.tolist(), tk_offset_valid.tolist())

            # Check that we have enough data

        mini_test(11, 8, 1, 1)
        mini_test(11, 8, 3, 1, gen_terminals_zeros)
        mini_test(11, 8, 3, 5)
        mini_test(11, 8, 5, 3, gen_terminals_zeros)
        mini_test(11, 8, 1, 1, gen_terminals_random)

if __name__ == '__main__':
    unittest.main()