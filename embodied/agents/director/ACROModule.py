import sys
import collections
import numpy as np
import embodied
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import behaviors
from . import nets
from . import tfagent
from . import tfutils


class FrameStack:
    def __init__(self, k):
        self.k = k
        self.deque = collections.deque(maxlen=k)

    def add_frame(self, frame):
        self.deque.append(frame)

    def get_frames(self):
        return tf.stack(list(self.deque), axis=1)


class ACROModule(tfutils.Module):

    # ACRO Model
    # Translation from World Model to ACRO space
    # SAR TODO ACRO to skill VAE

    def __init__(self, act_space, config):
        super().__init__()
        self.config = config

        # ACRO Model
        self.acro_state_size = self.img_size = (self.config.frame_stack,) + tuple(self.config.env.size) if self.config.env.gray else tuple(self.config.env.size) + (3,)

        self.acro_encoder_backbone = nets.MultiEncoder({"frame_stack": self.acro_state_size}, **self.config.acro_encoder_backbone)
        self.acro_embedding_head = nets.Dense(units=self.config.acro_embed_size, **self.config.acro_embedding_head)

        self.acro_action_head = nets.MLP(act_space.shape, **self.config.acro_action_head)

        # Translation Layer
        self.wm_to_acro_backbone = nets.MLP(shape=(self.config.acro_embed_size,), **self.config.wm_to_acro_backbone)

    def embed_acro(self, frame_stack):
        hidden = self.acro_encoder_backbone({"frame_stack": frame_stack})
        return self.acro_embedding_head(hidden)

    def translate_wm(self, wm_state):
        return self.wm_to_acro_backbone(wm_state)

    def train(self, data):
        # data is allegedly time-major ([T, batch, ...])
        images = data['image']
        actions = data['action']
        wm_states = data['wm_state'] # SAR TODO input this in training data

        # dimensions
        T = tf.shape(images)[0]
        batch = tf.shape(images)[1]
        k = self.config.acro_k_step
        frame_k = self.config.frame_stack

        fs_cur = FrameStack(frame_k) # current frame stack
        fs_fut = FrameStack(frame_k) # future frame stack, for k-step prediction loss
        metrics = {}

        # prefill frame stacks
        for t in range(frame_k):
            fs_cur.add_frame(images[t])
            fs_fut.add_frame(images[t + k])

        # slide along time
        for i in range(int(frame_k - 1), int(T - k)):
            acro_state_cur = self.embed_acro(fs_cur.get_frames())
            acro_state_fut = self.embed_acro(fs_fut.get_frames())

            true_action = actions[i]
            wm_state = wm_states[i]

            # calculate losses
            with tf.GradientTape() as translation_tape:
                pred_wm = self.translate_wm(wm_state)
                wm_loss = tf.reduce_mean(tf.square(pred_wm - acro_state_fut)) # MSE Loss

            with tf.GradientTape() as action_tape:
                pred_action = self.acro_action_head({'state_t': acro_state_cur, 'state_tk': acro_state_fut})
                norm_true = tf.nn.l2_normalize(true_action, axis=-1)
                action_loss = tf.reduce_mean(tf.square(pred_action - norm_true))

            # make updates
            metrics.update(self.opt(translation_tape, wm_loss, [self.wm_to_acro_backbone]))
            metrics.update(self.opt(action_tape, action_loss, [self.acro_action_backbone,self.acro_embedding_head, self.acro_encoder_backbone]))

            # update state
            next_cur = i + 1
            next_fut = i + k + 1
            fs_cur.add_frame(images[next_cur])
            fs_fut.add_frame(images[next_fut])

        return metrics
