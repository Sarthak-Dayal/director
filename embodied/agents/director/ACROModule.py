import sys
import numpy as np
import embodied
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import nets
from . import tfutils

class ACROModule(tfutils.Module):
    def __init__(self, act_space, config):
        super().__init__()
        self.config = config

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

        # Translation layer
        # self.wm_to_acro_backbone = nets.MLP(
        #     shape=(self.config.acro_embed_size,),
        #     **self.config.wm_to_acro_backbone
        # )
        # self.opt_wm = tfutils.Optimizer('acro_wm', **self.config.acro_opt)
        self.opt_act = tfutils.Optimizer('acro_action', **self.config.acro_opt)

    @tf.function
    def embed_acro(self, frame_stack):
        hidden = self.acro_encoder_backbone({"frame_stack": frame_stack})
        return self.acro_embedding_head(hidden)

    # def translate_wm(self, wm_state):
    #     return self.wm_to_acro_backbone(wm_state)

    @tf.function
    def _stack_frames(self, images, t):
        frame_k = tf.constant(self.config.frame_stack, dtype=tf.int32)
        start = t - frame_k + 1
        window = images[start : t + 1]  # [k, B, H, W, C]
        permuted = tf.transpose(window, perm=[1, 2, 3, 4, 0])  # [B, H, W, C, k]
        shape = tf.shape(permuted)
        B, H, W, C, K = shape[0], shape[1], shape[2], shape[3], shape[4]
        return tf.reshape(permuted, [B, H, W, C * K])

    @tf.function
    def train(self, data):
        # Unpack
        images = data['image']  # [T, B, H, W, C]
        actions = data['action']
        # stoch = data['wm_state']['stoch']
        # deter = data['wm_state']['deter']

        # Prepare WM states
        # flattened_stoch = tf.reshape(
        #     stoch,
        #     [tf.shape(stoch)[0], tf.shape(stoch)[1], -1]
        # )
        # wm_states = tf.concat([flattened_stoch, deter], axis=-1)

        # Loop bounds
        T = tf.shape(images)[0]
        k = tf.constant(self.config.acro_k_step, dtype=tf.int32)
        frame_k = tf.constant(self.config.frame_stack, dtype=tf.int32)

        # Static dims for shape enforcement
        size_H, size_W = self.config.env.size
        channels = 1 if self.config.env.gray else 3
        C_stack = channels * self.config.frame_stack

        # Calculate iteration count
        start_i = frame_k - 1
        end_i = T - k
        N = end_i - start_i

        # TensorArrays for metrics
        # ta_wm = tf.TensorArray(tf.float32, size=N)
        ta_action = tf.TensorArray(tf.float32, size=N)
        idx = tf.constant(0, tf.int32)

        # Loop in graph
        for i in tf.range(start_i, end_i):
            # Build frame stacks
            cur = self._stack_frames(images, i)
            cur = tf.ensure_shape(cur, [None, size_H, size_W, C_stack])
            fut = self._stack_frames(images, i + k)
            fut = tf.ensure_shape(fut, [None, size_H, size_W, C_stack])

            # wm_state = wm_states[i]

            # Translation update
            # with tf.GradientTape() as translation_tape:
            #     acro_cur = self.embed_acro(cur)
            #     acro_fut = self.embed_acro(fut)
                # wm_dist = self.translate_wm(wm_state)
                # acro_fut = tf.cast(acro_fut, wm_dist.dtype)
                # wm_loss = -tf.reduce_mean(wm_dist.log_prob(tf.cast(acro_cur, wm_dist.dtype)))
            
            # self.opt_wm(translation_tape, wm_loss, [self.wm_to_acro_backbone])

            # Action update
            with tf.GradientTape() as action_tape:
                acro_cur2 = self.embed_acro(cur)
                acro_fut2 = self.embed_acro(fut)
                action_dist = self.acro_action_head({
                    'state_t': acro_cur2,
                    'state_tk': acro_fut2
                })
                action_loss = -tf.reduce_mean(
                    action_dist.log_prob(actions[i])
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

            # Record losses
            # ta_wm = ta_wm.write(idx, wm_loss)
            ta_action = ta_action.write(idx, action_loss)
            idx += 1

        # Stack and return
        # wm_losses = ta_wm.stack()
        action_losses = ta_action.stack()
        return {'acro_action_loss': action_losses}
