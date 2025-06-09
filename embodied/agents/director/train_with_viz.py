import collections
import re
import warnings

import embodied
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from embodied.agents.director import behaviors
from embodied.replay import DiskStore


def train_with_viz(agent, env, train_replay, eval_replay, logger, args, run_tsne=False, reset_acro=False):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_train = embodied.when.Every(args.train_every)
  should_log = embodied.when.Every(args.log_every)
  should_expl = embodied.when.Until(args.expl_until)
  should_video = embodied.when.Every(args.eval_every)
  step = logger.step

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep):
    metrics = {}
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'Episode has {length} steps and return {score:.1f}.')
    metrics['length'] = length
    metrics['score'] = score
    metrics['reward_rate'] = (ep['reward'] - ep['reward'].min() >= 0.1).mean()
    logs = {}
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        logs[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        logs[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        logs[f'max_{key}'] = ep[key].max(0).mean()
    if should_video(step):
      for key in args.log_keys_video:
        if key == 'none':
          continue
        metrics[f'policy_{key}'] = ep[key]
        if 'log_goal' in ep:
          if ep['image'].shape == ep['log_goal'].shape:
            goal = (255 * ep['log_goal']).astype(np.uint8)
            metrics[f'policy_{key}_with_goal'] = np.concatenate(
                [ep['image'], goal], 2)
    logger.add(metrics, prefix='episode')
    logger.add(logs, prefix='logs')
    logger.add(train_replay.stats, prefix='replay')
    logger.write()

  fill = max(0, args.eval_fill - len(eval_replay))
  if fill:
    print(f'Fill eval dataset ({fill} steps).')
    eval_driver = embodied.Driver(env)
    eval_driver.on_step(eval_replay.add)
    random_agent = embodied.RandomAgent(env.act_space)
    eval_driver(random_agent.policy, steps=fill, episodes=1)
    del eval_driver

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(train_replay.add)
  fill = max(0, args.train_fill - len(train_replay))
  if fill:
    print(f'Fill train dataset ({fill} steps).')
    random_agent = embodied.RandomAgent(env.act_space)
    driver(random_agent.policy, steps=fill, episodes=1)

  dataset_train = iter(agent.dataset(train_replay.dataset))
  dataset_eval = iter(agent.dataset(eval_replay.dataset))
  state = [None]  # To be writable from train step function below.
  assert args.pretrain > 0  # At least one step to initialize variables.
  for _ in range(args.pretrain):
    _, state[0], _ = agent.train(next(dataset_train), state[0])

  metrics = collections.defaultdict(list)
  batch = [None]
  def train_step(tran, worker):
    if should_train(step):
      for _ in range(args.train_steps):
        batch[0] = next(dataset_train)
        outs, state[0], mets = agent.train(batch[0], state[0])
        [metrics[key].append(value) for key, value in mets.items()]
        if 'priority' in outs:
          train_replay.prioritize(outs['key'], outs['priority'])
    if should_log(step):
      with warnings.catch_warnings():  # Ignore empty slice warnings.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for name, values in metrics.items():
          logger.scalar('train/' + name, np.nanmean(values, dtype=np.float64))
          metrics[name].clear()
      logger.add(agent.report(batch[0]), prefix='report')
      logger.add(agent.report(next(dataset_eval)), prefix='eval')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
  checkpoint_acro = embodied.Checkpoint(logdir / 'checkpoint_acro.pkl')

  checkpoint.step = step
  # checkpoint.agent = agent
  checkpoint.wm = agent.agent.wm
  checkpoint.task_behavior = agent.agent.task_behavior
  checkpoint.expl_behavior = agent.agent.expl_behavior

  checkpoint_acro.acro_m = agent.agent.acro_m

  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  checkpoint.load_or_save()

  print("reached past checkpoint.load_or_save()")
  
  if reset_acro:
    acro_m = agent.agent.acro_m
    # Reset weights in acro_m without changing references
    print("Resetting weights in acro_m")

    if hasattr(acro_m, 'reset_weights'):
        acro_m.reset_weights()
    elif hasattr(acro_m, 'reset_parameters'):
        acro_m.reset_parameters()
    else:
        # Fallback: Reset each variable with appropriate initialization
        for var in acro_m.variables:
            dtype = var.dtype
            shape = var.shape

            # If the variable is not a floating‐point type, just zero it out.
            if not dtype.is_floating:
                var.assign(tf.zeros(shape, dtype=dtype))
                continue

            name = var.name.lower()
            if 'kernel' in name:
                # HeNormal is a common default for convolutional/dense kernels
                initializer = tf.keras.initializers.HeNormal()
                var.assign(initializer(shape, dtype=dtype))
            elif 'bias' in name:
                # Biases are usually initialized to zero
                var.assign(tf.zeros(shape, dtype=dtype))
            else:
                # For any other floating‐point variable (e.g., layer‐norm scales, etc.),
                # use a small uniform initializer in [-0.1, 0.1].
                initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
                var.assign(initializer(shape, dtype=dtype))


    print(f"Done resetting {len(list(acro_m.variables))} variables in acro_m.")
  
  if run_tsne:
    # Load up offline data
    disk = embodied.replay.DiskStore(logdir / 'episodes', parallel=False)

    # Encode a bunch of images and preserve the embeddings
    keys = disk.keys()
    embeds = []
    acro_k_step = 5
    frame_k = 3

    for key in list(keys)[:5]: # might want to limit this to a subset    
      # run through ACRO encoder
      print(f'Processing key: {key}')
      traj = disk[key]
      images = traj['image']

      images = tf.expand_dims(images, axis=1)  # Now shape is (T, 1, W, H, C)

      T = tf.shape(images)[0]
      k = tf.constant(acro_k_step, dtype=tf.int32)

      start_i = frame_k - 1
      end_i = T - k
      N = end_i - start_i

      for i in tf.range(start_i, end_i):
          cur = acro_m._stack_frames(images, i)
          acro_cur2 = acro_m.embed_acro(cur)
          embeds.append(acro_cur2.numpy())


    # Run t-SNE on the embeddings
    TSNE_model = TSNE(n_components=2, random_state=42)
    embeds_2d = TSNE_model.fit_transform(np.array(embeds).squeeze(axis=1))

    # Plot the results
    plt.figure(figsize=(8,8))
    plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1])
    plt.title('t-SNE of ACRO Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
    plt.savefig("tsne_results.png")
  else:
    print('Start training loop.')
    policy = lambda *args: agent.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    while step < args.steps:
      # scalars = collections.defaultdict(list)
      # for _ in range(args.eval_samples):
      #   for key, value in agent.report(next(dataset_eval)).items():
      #     if value.shape == ():
      #       scalars[key].append(value)
      # for name, values in scalars.items():
      #   logger.scalar(f'eval/{name}', np.array(values, np.float64).mean())
      logger.write()
      driver(policy, steps=args.eval_every)
      checkpoint.save()
      checkpoint_acro.save()

