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

from umap import UMAP
from embodied.envs.n_rooms import NRooms
from embodied.agents.director.umap_settings import *


def train_with_viz(agent, env, train_replay, eval_replay, logger, args, run_tsne=False, run_umap=False, reset_acro=False):

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
  checkpoint_acro.load_or_save()

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
  elif run_umap:
    n_neighbors = 15
    min_dist = 0.1
    metric = 'euclidean'
    

    umap_configuration = CORRESPONDING_POS

    layout_str = umap_configuration.layout_str

    # Points are stored as pos, goal
    structured_data = umap_configuration.structured_data

    images = []
    colors = []
    category_labels = umap_configuration.category_labels
    point_categories = []
    category_images = [[] for _ in range(len(structured_data))]  # Store images by category
    
    for i, category in enumerate(structured_data):
      for sample in category:
        img = NRooms.render_pure(sample[0], sample[1], layout_str)
        images.append(img)
        category_images[i].append(img)
        point_categories.append(i)
        
        if umap_configuration.color_setting == "gradient":
            # Interpolate between red and blue based on category index
            # First category (i=0) will be fully red, last category will be fully blue
            if len(structured_data) > 1:
                # Normalize i to range [0, 1]
                t = i / (len(structured_data) - 1)
                # Red component decreases from 1 to 0
                r = 1.0 - t
                # Blue component increases from 0 to 1
                b = t
                colors.append([r, 0.0, b])
            else:
                # If there's only one category, make it red
                colors.append([1.0, 0.0, 0.0])
        else:
            # Original random color generation
            rng = np.random.default_rng(1234 + i)
            colors.append(rng.random(3))
    
    images = tf.convert_to_tensor(np.array(images))
    acro_m = agent.agent.acro_m
    acro_embed = acro_m.embed_acro(images)
    
    fit = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric
    )
    u = fit.fit_transform(acro_embed)
    
    # Create a figure with two subplots: one for UMAP and one for the image grid
    fig = plt.figure(figsize=(18, 12))
    
    # 1. UMAP plot
    ax1 = fig.add_subplot(121)
    
    # Create a scatter plot for each category
    for i, label in enumerate(category_labels):
        category_points = [j for j, cat in enumerate(point_categories) if cat == i]
        if category_points:
            category_colors = [colors[j] for j in category_points]
            category_embeddings = u[category_points]
            ax1.scatter(category_embeddings[:, 0], category_embeddings[:, 1], 
                        color=colors[category_points[0]], label=label)
    
    ax1.set_title('UMAP Projection of ACRO Embeddings')
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    # ax1.legend()
    
    # 2. Image grid
    # ax2 = fig.add_subplot(122)
    # ax2.axis('off')
    
    # # Limit to 4 images per category
    # images_per_category = 4
    # n_categories = len(category_images)
    # grid_height = n_categories
    # grid_width = images_per_category
    
    # # Create grid of images by category
    # grid = np.ones((grid_height * 64, grid_width * 64, 3))  # Assuming 64x64 images
    
    # for i, cat_imgs in enumerate(category_images):
    #     # Only use up to 4 images per category
    #     for j, img in enumerate(cat_imgs[:images_per_category]):
    #         # Place each image in the grid
    #         if len(img.shape) == 3:  # Check if the image has a color channel
    #             grid[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = img
    #         else:  # If it's grayscale, repeat the channel
    #             grid[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = np.stack([img, img, img], axis=-1)
                
    # # Display the grid
    # ax2.imshow(grid)
    
    # # Add category labels to the left side of each row
    # for i, label in enumerate(category_labels):
    #     ax2.text(-5, i * 64 + 32, label, horizontalalignment='right', 
    #             verticalalignment='center', fontsize=10)
        
    # ax2.set_title('Image Samples by Category')
    plt.tight_layout()
    plt.savefig("umap_results.png", dpi=300)
    
    # Create a second figure just for the image grid with more detailed labels
    fig2, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Create a new grid with only 4 images per category for the second figure
    # grid2 = np.ones((grid_height * 64, grid_width * 64, 3))  # Using the same dimensions as before
    
    # for i, cat_imgs in enumerate(category_images):
    #     # Only use up to 4 images per category
    #     for j, img in enumerate(cat_imgs[:images_per_category]):
    #         # Place each image in the grid
    #         if len(img.shape) == 3:  # Check if the image has a color channel
    #             grid2[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = img
    #         else:  # If it's grayscale, repeat the channel
    #             grid2[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = np.stack([img, img, img], axis=-1)
    
    # # Display the image grid with more space
    # # ax.imshow(grid2)
    
    # # Add category labels as row headers with more visibility
    # for i, label in enumerate(category_labels):
    #     ax.text(-10, i * 64 + 32, label, horizontalalignment='right', 
    #             verticalalignment='center', fontsize=12, fontweight='bold')
    
    # plt.tight_layout()
    # plt.savefig("image_grid.png", dpi=300)
    
    import pdb; pdb.set_trace()
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

