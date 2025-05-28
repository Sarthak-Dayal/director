# train_acro.py
import sys
import os

import pathlib

directory = pathlib.Path(__file__)

directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
sys.path.append(str(directory.parent.parent))
__package__ = directory.name

import pathlib
import warnings

import numpy as np
import tensorflow as tf
import wandb
import embodied
from ACROModule import ACROModule

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')

def main(argv=None):
    # ——— 1) Parse flags and build `config` ——————————————————————————
    # We only pull in the parts of the agent‐config that ACROModule actually uses.
    from . import agent as agnt

    parsed, other = embodied.Flags(
        configs = ['defaults'],    # override via --configs foo bar
        npz_dir = '/datastor1/sarthakd',            # e.g. --npz_dir=/path/to/episodes
        logdir  = '/datastor1/sarthakd/ACRO_OFF_Logdir',        # e.g. --logdir=/datastor1/.../ACRO_OFF_Logdir
    ).parse_known(argv)

    # base config from agent’s defaults
    config = embodied.Config(agnt.Agent.configs['defaults'])

    for name in parsed.configs:
        config = config.update(agnt.Agent.configs[name])
    # parse any other overrides
    config = embodied.Flags(config).parse(other)

    config = config.update(logdir = parsed.logdir)
    config = config.update(npz_dir = parsed.npz_dir)
    
    # cast logdir & npz_dir to Path
    logdir  = embodied.Path(config.logdir)
    npz_dir = pathlib.Path(parsed.npz_dir)
    logdir.mkdirs()

    # ——— 2) Set up W&B & TensorBoard ————————————————————————————
    wandb.init(
        project="Director ACRO",
        sync_tensorboard=True,
        reinit=True,
        dir="/datastor1/sarthakd/wandb"
    )

    tb_writer = tf.summary.create_file_writer(str(logdir / "tensorboard"))

    # ——— 3) Instantiate ACROModule (action‐only) —————————————————————
    # dummy B=1 batch for shape inference
    dummy_action_shape = (1,) + tuple([21])
    acro = ACROModule({'action': dummy_action_shape}, config)

    # ——— 4) Loop over all .npz files under npz_dir ——————————————————
    npz_files = list(npz_dir.rglob("*.npz"))
    for npz_path in npz_files:
        print(f"▶ loading {npz_path}")
        data = np.load(str(npz_path), allow_pickle=True)
        if 'image' not in data or 'action' not in data:
            continue
        images = data['image']   # [T, H, W, C]
        actions = data['action'] # [T, …]

        # add batch dim
        T, H, W, C = images.shape
        images = images.reshape(T, 1, H, W, C)
        actions = actions.reshape(T, 1, *actions.shape[1:])

        images_tf  = tf.convert_to_tensor(images,  dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)

        # ——— 5) Run action‐only train() and grab losses ——————————————
        out = acro.train({
            'image':  images_tf,
            'action': actions_tf
        })
        losses = out['acro_action_loss'].numpy()  # [num_steps]

        # ——— 6) Log each step’s loss to TB & W&B ——————————————————————
        for step, loss in enumerate(losses):
            with tb_writer.as_default():
                tf.summary.scalar("acro_action_loss", loss, step=step)
            wandb.log({
                "acro_action_loss": loss,
                "npz_file": npz_path.name
            }, step=step)

    tb_writer.flush()
    tb_writer.close()
    wandb.finish()

if __name__ == "__main__":
    main()
