import fileinput
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import OneHotCategoricalStraightThrough
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.distributions as td
from tqdm import tqdm

from embodied.agents.director.tfutils import Optimizer, OneHotDist

# Fix __file__ issue
try:
    directory = pathlib.Path(__file__).resolve()
except NameError:
    directory = pathlib.Path.cwd()  # Fallback if __file__ is not available

# Ensure correct module resolution
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))

try:
    import google3  # noqa
except ImportError:
    pass  # Ignore if google3 is not available

# Import local modules correctly
import embodied
import embodied.agents.director.agent_torch as agnt  # FIX: Avoid relative imports
import embodied.agents.director.nets_torch as nets_torch


def analyze_model_gradients(model):
    """
    Prints named parameters and their gradient status,
    then zeros the gradients.

    Args:
        model: PyTorch model (nn.Module)
    """
    print("Analyzing model parameter gradients:")
    print("-" * 50)

    # Iterate through named parameters
    for name, param in model.named_parameters():
        if param.grad is None:
            status = "None (no gradient computed)"
        elif torch.all(param.grad == 0):
            status = "All zeros"
        else:
            status = "Non-zero"
            # Count number of zero and non-zero elements in gradient
            num_zeros = torch.sum(param.grad == 0).item()
            num_elements = param.grad.numel()
            percent_zeros = (num_zeros / num_elements) * 100

            status += f" ({percent_zeros:.2f}% zeros)"

        # Get parameter and gradient shape
        param_shape = str(tuple(param.shape))
        grad_shape = str(tuple(param.grad.shape)) if param.grad is not None else "N/A"

        print(f"Parameter: {name}")
        print(f"Parameter Shape: {param_shape}")
        print(f"Gradient Shape: {grad_shape}")
        print(f"Gradient Status: {status}")
        print(f"Requires grad: {param.requires_grad}")
        print("-" * 50)

    # Zero the gradients
    # model.zero_grad()
    # print("All gradients have been zeroed.")


def analyze_tensor_gradients(tensor_dict, prefix=""):
    """
    Recursively analyze the gradients of tensors in a nested dictionary.

    For each tensor found, the function checks its .grad attribute:
      - If grad is None, it prints "None"
      - If grad exists but is all zeros, it prints "All Zeros"
      - Otherwise, it prints summary statistics (mean, std, min, max)

    Args:
        tensor_dict (dict): A dictionary where some values are torch.Tensors.
        prefix (str): A prefix used to build hierarchical keys.
    """
    for key, value in tensor_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            analyze_tensor_gradients(value, prefix=full_key)
        elif torch.is_tensor(value):
            grad = getattr(value, "grad", None)
            if grad is None:
                status = "\033[93mNone\033[0m"
                print(f"{full_key:<50} : Grad = {status}")
            elif torch.all(grad == 0):
                status = "\033[91mAll Zeros\033[0m"
                print(f"{full_key:<50} : Grad = {status}")
            else:
                # Convert to float for statistics if necessary.
                grad_float = grad.float() if not torch.is_floating_point(grad) else grad
                stats = {
                    "mean": grad_float.mean().item(),
                    "std": grad_float.std().item(),
                    "min": grad_float.min().item(),
                    "max": grad_float.max().item(),
                }
                print(f"{full_key:<50} : Grad = \033[92mOK\033[0m")
                print(f"{'':<52}   mean: {stats['mean']:.4e}, std: {stats['std']:.4e}, "
                      f"min: {stats['min']:.4e}, max: {stats['max']:.4e}")
        else:
            # If not a tensor, optionally skip or note the type.
            print(f"{full_key:<50} : \033[90m({type(value).__name__})\033[0m - Skipped")

def enable_grad_tracking(x):
    if isinstance(x, torch.Tensor):
        if not x.requires_grad:
            x.requires_grad_()
        x.retain_grad()
        return x
    elif isinstance(x, dict):
        return {k: enable_grad_tracking(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(enable_grad_tracking(v) for v in x)
    else:
        return x  # Leave other types untouched

def test_image_vae():
    parsed, other = embodied.Flags(
        configs=['defaults'], actor_id=0, actors=0,
    ).parse_known(None)
    config = embodied.Config(agnt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agnt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)

    shapes = {'image': (64, 64, 3)}

    encoder = nets_torch.MultiEncoder(shapes, **config.encoder)
    latent = nets_torch.MLP((1024,), **config.goal_encoder)
    config = config.update({"decoder": config.decoder.update({'inputs' : ('embed',)})})
    decoder = nets_torch.MultiDecoder(shapes, **config.decoder)

    batch_size = 1024
    prior_logits = torch.zeros(batch_size, 1024).cuda()
    # OneHotDist is defined in tfutils.
    prior = OneHotDist(prior_logits)
    # Wrap with Independent: note that if shape has more than one dim, use len(shape)-1 event dims.
    prior = td.Independent(prior, 1)

    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize model, optimizer
    opt = Optimizer('model', **config.model_opt)

    # Visualize reconstruction
    def visualize_reconstruction():
        images, _ = next(iter(dataloader))
        images = images.cuda()
        images = F.interpolate(images, size=(64, 64), mode='bilinear')
        images = {'image': images.reshape(-1, 64, 64, 3)}
        # images = images.cuda()
        embed = encoder(images)
        z = td.Independent(latent({'goal': embed}), reinterpreted_batch_ndims=1)
        rec = decoder({'embed': embed})['image']

        recon_images = rec.mode

        fig, axes = plt.subplots(4, 4, figsize=(11, 11))
        for i in range(8):
            axes[0 + 2 * (i // 4), i % 4].imshow(np.transpose(images['image'][i].cpu().detach().numpy(), (0, 1, 2)))
            axes[1 + 2 * (i // 4), i % 4].imshow(np.transpose(recon_images[i].cpu().detach().numpy(), (0, 1, 2)))
            axes[0 + 2 * (i // 4), i % 4].axis('off')
            axes[1 + 2 * (i // 4), i % 4].axis('off')
        plt.show()

    # Training loop
    num_epochs = 100
    loss = torch.tensor(0.0)
    for epoch in range(num_epochs):
        for images, _ in tqdm(dataloader, desc="Processing Batches"):
            images = images.cuda()
            images = F.interpolate(images, size=(64, 64), mode='bilinear')
            images = {'image': images.reshape(-1, 64, 64, 3)}
            # images = images.cuda()
            embed = encoder(images)
            z = td.Independent(latent({'goal': embed}), reinterpreted_batch_ndims=1)
            rec = decoder({'embed': embed})['image']

            rec_loss = -rec.log_prob(images['image']).mean()
            # kl_loss = td.kl_divergence(z, prior).mean()

            loss = rec_loss # + kl_loss

            modules = [encoder, latent, decoder]
            opt.step(loss, modules)

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        visualize_reconstruction()

def test_image_vae_no_decoder():
    parsed, other = embodied.Flags(
        configs=['defaults'], actor_id=0, actors=0,
    ).parse_known(None)
    config = embodied.Config(agnt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agnt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)

    shapes = {'image': (64, 64, 3)}

    encoder = nets_torch.MultiEncoder(shapes, **{'mlp_keys': '$^',
                                                 'cnn_keys': 'image',
                                                 'act': 'elu',
                                                 'norm': 'layer',
                                                 'mlp_layers': 4,
                                                 'mlp_units': 512,
                                                 'cnn': 'simple',
                                                 'cnn_depth': 16,
                                                 'cnn_kernels': (4, 4, 4, 4)})
    latent = nets_torch.MLP(None, **{'layers': 1,
                                             'units': 512,
                                             'act': 'elu',
                                             'norm': 'layer',
                                             'dist': 'onehot',
                                             'outscale': 0.1,
                                             'unimix': 0.0,
                                             'inputs': ('goal',)})
    config = config.update({"decoder": config.decoder.update({'inputs' : ('embed',)})})
    decoder = nets_torch.MultiDecoder(shapes, **{'mlp_keys': '$^',
                                                 'cnn_keys': 'image',
                                                 'act': 'elu',
                                                 'norm': 'layer',
                                                 'mlp_layers': 4,
                                                 'mlp_units': 512,
                                                 'cnn': 'simple',
                                                 'cnn_depth': 64,
                                                 'cnn_kernels': (5, 5, 6, 6),
                                                 'image_dist': 'mse',
                                                 'inputs': ('embed',)})
    decoder_baseline =  nn.Sequential(
                            nn.LazyConvTranspose2d( 64, kernel_size=(5, 5), stride=(2, 2)),
                            nn.ReLU(),
                            nn.LazyConvTranspose2d(32, kernel_size=(5, 5), stride=(2, 2)),
                            nn.ReLU(),
                            nn.LazyConvTranspose2d(16, kernel_size=(6, 6), stride=(2, 2)),
                            nn.ReLU(),
                            nn.LazyConvTranspose2d(3, kernel_size=(6, 6), stride=(2, 2)),
                            nn.Sigmoid()
                        ).cuda()

    batch_size = 1024
    prior_logits = torch.zeros(batch_size, 512).cuda()
    # OneHotDist is defined in tfutils.
    prior = OneHotDist(prior_logits)
    # Wrap with Independent: note that if shape has more than one dim, use len(shape)-1 event dims.
    # prior = td.Independent(prior, 1)

    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize model, optimizer
    opt = Optimizer('model', **config.model_opt)

    # Visualize reconstruction
    def visualize_reconstruction():
        images, _ = next(iter(dataloader))
        images = images.cuda()
        images = F.interpolate(images, size=(64, 64), mode='bilinear')
        images = {'image': images.reshape(-1, 64, 64, 3)}
        # images = images.cuda()
        embed = encoder(images)
        z = latent({'goal': embed})
        dist = td.Independent(OneHotCategoricalStraightThrough(logits=z), reinterpreted_batch_ndims=1)
        # z = td.Independent(latent({'goal': embed}), reinterpreted_batch_ndims=1)
        # rec = decoder({'embed': embed})['image']
        # recon_images = rec.mode

        rec = decoder_baseline(dist.sample().view(-1, 512, 1, 1))
        recon_images = rec

        fig, axes = plt.subplots(4, 4, figsize=(11, 11))
        for i in range(8):
            axes[0 + 2 * (i // 4), i % 4].imshow(np.transpose(images['image'][i].cpu().detach().numpy(), (0, 1, 2)))
            axes[1 + 2 * (i // 4), i % 4].imshow(np.transpose(recon_images[i].cpu().detach().numpy(), (1, 2, 0)))
            axes[0 + 2 * (i // 4), i % 4].axis('off')
            axes[1 + 2 * (i // 4), i % 4].axis('off')
        plt.show()

    # Training loop
    num_epochs = 100
    loss = torch.tensor(0.0)
    for epoch in range(num_epochs):
        for images, _ in tqdm(dataloader, desc="Processing Batches"):
            images = images.cuda()
            images = F.interpolate(images, size=(64, 64), mode='bilinear')
            images = {'image': images.reshape(-1, 64, 64, 3)}
            # images = images.cuda()
            embed = encoder(images)
            z = latent({'goal': embed})
            dist = OneHotCategoricalStraightThrough(logits=z)
            # z = td.Independent(latent({'goal': embed}), reinterpreted_batch_ndims=1)
            # rec = decoder({'embed': embed})['image']
            # rec_loss = -rec.log_prob(images['image']).mean()
            rec = decoder_baseline(dist.rsample().view(-1, 512, 1, 1))
            rec_loss = 100 * torch.nn.functional.mse_loss(rec.permute(0, 2, 3, 1), images['image'], reduction='mean')

            kl_loss = td.kl_divergence(dist, prior).mean()

            loss = rec_loss + kl_loss

            modules = [encoder, latent, decoder, decoder_baseline]
            opt.step(loss, modules)

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        visualize_reconstruction()



if __name__ == '__main__':
    test_image_vae_no_decoder()