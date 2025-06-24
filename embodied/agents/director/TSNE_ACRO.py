import sys, pathlib

directory = pathlib.Path(__file__)
try:
  import google3  # noqa
except ImportError:
  directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf

logdir = "/datastor1/sarthakd/logdir/20250528-183819"
logdir = embodied.Path(logdir)

def tsne_acro(agent, train_replay, eval_replay, logger):
  # Load up ACRO checkpoint
  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
  checkpoint.step = logger.step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  checkpoint.load()
  agent = checkpoint._values['agent'].agent
  acro_m = agent.acro_m
  # Load up offline data
  disk = embodied.replay.DiskStore(logdir / 'episodes', parallel=False)

  # Encode a bunch of images and preserve the embeddings
  keys = disk.keys()
  embeds = []
  acro_k_step = 5
  frame_k = 3

  for key in keys: # might want to limit this to a subset    
    # run through ACRO encoder

    traj = disk[key]
    images = traj['image']

    T = tf.shape(images)[0]
    k = tf.constant(acro_k_step, dtype=tf.int32)

    start_i = frame_k - 1
    end_i = T - k
    N = end_i - start_i

    for i in tf.range(start_i, end_i):
        cur = acro_m._stack_frames(images, i)
        acro_cur2 = acro_m.embed_acro(cur)
        embeds.append(acro_cur2.detach().cpu().numpy())


  # Run t-SNE on the embeddings
  TSNE_model = TSNE(n_components=2, random_state=42)
  embeds_2d = TSNE_model.fit_transform(embeds)

  # Plot the results
  plt.figure(figsize=(8,8))
  plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1])
  plt.title('t-SNE of ACRO Embeddings')
  plt.xlabel('Component 1')
  plt.ylabel('Component 2')
  plt.show()