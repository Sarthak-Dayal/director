# nets_torch.py
import functools
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

# Import our common helper functions and classes from tfutils
from tfutils import Module, SymlogDist, MSEDist, scan, map_structure, get_act, Input, tensor, symlog, symexp

# ---------------------------
# RSSM
# ---------------------------
class RSSM(Module):
    def __init__(self, deter=1024, stoch=32, classes=32, unroll=True, initial='zeros', **kw):
        super().__init__()
        self._deter = deter
        self._stoch = stoch
        self._classes = classes
        self._unroll = unroll
        self._initial = initial
        self._kw = kw
        # Use the default dtype as set in torch
        self._cast = lambda x: x.to(torch.get_default_dtype())

    def initial(self, batch_size):
        dtype = torch.get_default_dtype()
        if self._classes:
            state = {
                'deter': torch.zeros(batch_size, self._deter, dtype=dtype),
                'logit': torch.zeros(batch_size, self._stoch, self._classes, dtype=dtype),
                'stoch': torch.zeros(batch_size, self._stoch, self._classes, dtype=dtype)
            }
        else:
            state = {
                'deter': torch.zeros(batch_size, self._deter, dtype=dtype),
                'mean': torch.zeros(batch_size, self._stoch, dtype=dtype),
                'std': torch.ones(batch_size, self._stoch, dtype=dtype),
                'stoch': torch.zeros(batch_size, self._stoch, dtype=dtype)
            }
        if self._initial == 'zeros':
            return state
        elif self._initial == 'learned':
            init_deter = self._cast(self.get('initial_deter', nn.Parameter, state['deter'][0].float(), requires_grad=True))
            init_stoch = self._cast(self.get('initial_stoch', nn.Parameter, state['stoch'][0].float(), requires_grad=True))
            state['deter'] = init_deter.unsqueeze(0).repeat(batch_size, 1)
            state['stoch'] = init_stoch.unsqueeze(0).repeat(batch_size, 1)
            return state
        elif self._initial == 'learned2':
            init_deter = torch.tanh(self.get('initial_deter', nn.Parameter, state['deter'][0].float(), requires_grad=True))
            state['deter'] = self._cast(init_deter.unsqueeze(0).repeat(batch_size, 1))
            state['stoch'] = self.get_stoch(state['deter'])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None, training=False):
        swap = lambda x: x.permute([1, 0] + list(range(2, x.dim())))
        if state is None:
            state = self.initial(action.shape[0])
        # Use a simple step function that calls obs_step
        step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
        inputs = (swap(action), swap(embed), swap(is_first))
        start = (state, state)
        post, prior = scan(step, inputs, start, static=self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None, training=False):
        swap = lambda x: x.permute([1, 0] + list(range(2, x.dim())))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = scan(self.img_step, action, state, static=self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_dist(self, state, argmax=False):
        if self._classes:
            logit = state['logit'].to(torch.float32)
            dist_obj = td.Independent(self.get('OneHotDist', td.OneHotCategorical, logits=logit), 1)
        else:
            mean, std = state['mean'].to(torch.float32), state['std'].to(torch.float32)
            dist_obj = td.Independent(td.Normal(mean, std), 1)
        return dist_obj

    def obs_step(self, prev_state, prev_action, embed, is_first):
        prev_state = map_structure(self._cast, prev_state)
        prev_action = self._cast(prev_action)
        is_first = self._cast(is_first)
        prev_state = map_structure(lambda x: torch.einsum('b...,b->b...', x, 1.0 - is_first), prev_state)
        init = self.initial(len(is_first))
        prev_state = {k: v + torch.einsum('b...,b->b...', self._cast(init[k]), is_first)
                      for k, v in prev_state.items()}
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self.get('obs_out', Dense, **self._kw)(x)
        stats = self._stats_layer('obs_stats', x)
        dist_obj = self.get_dist(stats)
        stoch = self._cast(dist_obj.sample())
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action):
        prev_stoch = self._cast(prev_state['stoch'])
        prev_action = self._cast(prev_action)
        if self._classes:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._classes]
            prev_stoch = prev_stoch.view(*shape)
        if prev_action.dim() > prev_stoch.dim():
            shape = list(prev_action.shape[:-2]) + [np.prod(prev_action.shape[-2:])]
            prev_action = prev_action.view(*shape)
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.get('img_in', Dense, **self._kw)(x)
        x, deter = self._gru(x, prev_state['deter'])
        x = self.get('img_out', Dense, **self._kw)(x)
        stats = self._stats_layer('img_stats', x)
        dist_obj = self.get_dist(stats)
        stoch = self._cast(dist_obj.sample())
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self.get('img_out', Dense, **self._kw)(deter)
        stats = self._stats_layer('img_stats', x)
        dist_obj = self.get_dist(stats)
        return self._cast(dist_obj.mode())

    def _gru(self, x, deter):
        x = torch.cat([deter, x], dim=-1)
        kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
        x = self.get('gru', Dense, **kw)(x)
        reset, cand, update = torch.chunk(x, 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def _stats_layer(self, name, x):
        if self._classes:
            x = self.get(name, Dense, self._stoch * self._classes)(x)
            new_shape = list(x.shape[:-1]) + [self._stoch, self._classes]
            logit = x.view(*new_shape)
            return {'logit': logit}
        else:
            x = self.get(name, Dense, 2 * self._stoch)(x)
            mean, std = torch.chunk(x, 2, dim=-1)
            std = 2 * torch.sigmoid(std / 2) + 0.1
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, balance=0.8):
        post_const = map_structure(lambda x: x.detach(), post)
        prior_const = map_structure(lambda x: x.detach(), prior)
        lhs = td.kl_divergence(self.get_dist(post_const), self.get_dist(prior))
        rhs = td.kl_divergence(self.get_dist(post), self.get_dist(prior_const))
        return balance * lhs + (1 - balance) * rhs

# ---------------------------
# MultiEncoder
# ---------------------------
class MultiEncoder(Module):
    def __init__(self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4, mlp_units=512, cnn='simple', cnn_depth=48,
                 cnn_kernels=(4, 4, 4, 4), cnn_blocks=2, **kw):
        # Exclude keys.
        super().__init__()
        excluded = ('is_first', 'is_last')
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if re.match(cnn_keys, k) and len(v)==3}
        self.mlp_shapes = {k: v for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) in (0,1)}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)
        if cnn == 'simple':
            self._cnn = ImageEncoderSimple(cnn_depth, cnn_kernels, **kw)
        else:
            raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, dist='none', **kw)
        self._cast = lambda x: x.to(torch.get_default_dtype())

    def forward(self, data):
        # Determine batch dims from one key.
        some_key, some_shape = list(self.shapes.items())[0]
        batch_dims = data[some_key].shape[:-len(some_shape)]
        data = {k: v.view(-1, *v.shape[len(batch_dims):]) for k, v in data.items()}
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([data[k] for k in self.cnn_shapes.keys()], dim=-1)
            output = self._cnn(inputs)
            output = output.view(output.shape[0], -1)
            outputs.append(output)
        if self.mlp_shapes:
            inputs = [data[k].unsqueeze(-1) if len(self.shapes[k])==0 else data[k]
                      for k in self.mlp_shapes.keys()]
            inputs = torch.cat([self._cast(x) for x in inputs], dim=-1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, dim=-1)
        new_shape = batch_dims + outputs.shape[1:]
        outputs = outputs.view(*new_shape)
        return outputs

# ---------------------------
# MultiDecoder
# ---------------------------
class MultiDecoder(Module):
    def __init__(self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4, mlp_units=512,
                 cnn='simple', cnn_depth=48, cnn_kernels=(5, 5, 6, 6), image_dist='mse', **kw):
        super().__init__()
        excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if re.match(cnn_keys, k) and len(v)==3}
        self.mlp_shapes = {k: v for k, v in shapes.items() if re.match(mlp_keys, k) and len(v)==1}
        self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
        print('Decoder CNN shapes:', self.cnn_shapes)
        print('Decoder MLP shapes:', self.mlp_shapes)
        if self.cnn_shapes:
            shapes_list = list(self.cnn_shapes.values())
            assert all(x[:-1] == shapes_list[0][:-1] for x in shapes_list)
            merged = shapes_list[0][:-1] + (sum(x[-1] for x in shapes_list),)
            if cnn == 'simple':
                self._cnn = ImageDecoderSimple(merged, cnn_depth, cnn_kernels, **kw)
            else:
                raise NotImplementedError(cnn)
        if self.mlp_shapes:
            self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, **kw)
        self._inputs = Input(inputs)
        self._image_dist = image_dist

    def forward(self, inputs):
        features = self._inputs(inputs)
        dists = {}
        if self.cnn_shapes:
            flat = features.view(-1, features.shape[-1])
            output = self._cnn(flat)
            output = output.view(*features.shape[:-1], output.shape[-1])
            splits = [v[-1] for v in self.cnn_shapes.values()]
            means = torch.split(output, splits, dim=-1)
            for key, mean in zip(self.cnn_shapes.keys(), means):
                dists[key] = self._make_image_dist(key, mean)
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, name, mean):
        mean = mean.to(torch.float32)
        if self._image_dist == 'normal':
            return td.Independent(td.Normal(mean, 1), 3)
        if self._image_dist == 'mse':
            return MSEDist(mean, 3, 'sum')
        raise NotImplementedError(self._image_dist)

# ---------------------------
# ImageEncoderSimple
# ---------------------------
class ImageEncoderSimple(Module):
    def __init__(self, depth, kernels, **kw):
        super().__init__()
        self._depth = depth
        self._kernels = kernels
        self._kw = kw

    def forward(self, features):
        x = features.to(torch.get_default_dtype())
        depth = self._depth
        for i, kernel in enumerate(self._kernels):
            x = self.get(f'conv{i}', Conv2D, depth, kernel, stride=2, pad='valid', **self._kw)(x)
            depth *= 2
        return x

# ---------------------------
# ImageDecoderSimple
# ---------------------------
class ImageDecoderSimple(Module):
    def __init__(self, shape, depth, kernels, **kw):
        super().__init__()
        self._shape = shape
        self._depth = depth
        self._kernels = kernels
        self._kw = kw

    def forward(self, features):
        x = features.to(torch.get_default_dtype())
        x = x.view(-1, 1, 1, x.shape[-1])
        depth = self._depth * (2 ** (len(self._kernels) - 2))
        for i, kernel in enumerate(self._kernels[:-1]):
            x = self.get(f'conv{i}', Conv2D, depth, kernel, transp=True, stride=2, pad='valid', **self._kw)(x)
            depth //= 2
        x = self.get('out', Conv2D, self._shape[-1], self._kernels[-1], transp=True)(x)
        x = torch.sigmoid(x)
        assert x.shape[-3:] == self._shape, f"{x.shape[-3:]} vs {self._shape}"
        return x

# ---------------------------
# MLP
# ---------------------------
class MLP(Module):
    def __init__(self, shape, layers, units, inputs=['tensor'], dims=None, **kw):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._layers = layers
        self._units = units
        self._inputs = Input(inputs, dims=dims)
        distkeys = ('dist', 'outscale', 'minstd', 'maxstd', 'unimix', 'outnorm')
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}
        self._dist = {k: v for k, v in kw.items() if k in distkeys}

    def forward(self, inputs):
        feat = self._inputs(inputs)
        x = feat.to(torch.get_default_dtype())
        x = x.view(-1, x.shape[-1])
        for i in range(self._layers):
            x = self.get(f'dense{i}', Dense, self._units, **self._dense)(x)
        x = x.view(*feat.shape[:-1], x.shape[-1])
        if self._shape is None:
            return x
        elif isinstance(self._shape, tuple):
            return self._out('out', self._shape, x)
        elif isinstance(self._shape, dict):
            return {k: self._out(k, v, x) for k, v in self._shape.items()}
        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x):
        return self.get(f'dist_{name}', DistLayer, shape, **self._dist)(x)

# ---------------------------
# DistLayer
# ---------------------------
class DistLayer(Module):
    def __init__(self, shape, dist='mse', outscale=0.1, minstd=0.1, maxstd=1.0, unimix=0.0):
        super().__init__()
        assert all(isinstance(dim, int) for dim in shape), shape
        self._shape = shape
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self._outscale = outscale
        self._unimix = unimix

    def forward(self, inputs):
        dist_obj = self.inner(inputs)
        return dist_obj

    def inner(self, inputs):
        kw = {}
        if self._outscale == 0.0:
            kw['weight_init'] = 'zeros'
        else:
            kw['weight_init'] = 'variance_scaling'
        out = self.get('out', Dense, int(np.prod(self._shape)), **kw)(inputs)
        new_shape = list(inputs.shape[:-1]) + list(self._shape)
        out = out.view(*new_shape).to(torch.float32)
        if self._dist == 'symlog':
            return SymlogDist(out, len(self._shape), 'sum')
        if self._dist == 'mse':
            return MSEDist(out, len(self._shape), 'sum')
        if self._dist == 'cos':
            assert len(self._shape) == 1
            return td.Independent(td.Normal(out, 1), 1)
        if self._dist == 'dir':
            lo, hi = self._minstd, self._maxstd
            std = self.get('std', Dense, int(np.prod(self._shape)))(inputs)
            std = std.view(*inputs.shape[:-1], *self._shape).to(torch.float32)
            std = (hi - lo) * torch.sigmoid(std) + lo
            dist_obj = td.Independent(td.Normal(out, std), 1)
            dist_obj.minent = np.prod(self._shape) * td.Normal(0.0, lo).entropy()
            dist_obj.maxent = np.prod(self._shape) * td.Normal(0.0, hi).entropy()
            return dist_obj
        if self._dist == 'normal':
            lo, hi = self._minstd, self._maxstd
            std = self.get('std', Dense, int(np.prod(self._shape)))(inputs)
            std = std.view(*inputs.shape[:-1], *self._shape).to(torch.float32)
            std = (hi - lo) * torch.sigmoid(std) + lo
            dist_obj = td.Normal(torch.tanh(out), std)
            dist_obj = td.Independent(dist_obj, len(self._shape))
            dist_obj.minent = np.prod(self._shape) * td.Normal(0.0, lo).entropy()
            dist_obj.maxent = np.prod(self._shape) * td.Normal(0.0, hi).entropy()
            return dist_obj
        if self._dist == 'binary':
            dist_obj = td.Bernoulli(logits=out)
            return td.Independent(dist_obj, len(self._shape))
        if self._dist == 'trunc_normal':
            lo, hi = self._minstd, self._maxstd
            std = self.get('std', Dense, int(np.prod(self._shape)))(inputs)
            std = std.view(*inputs.shape[:-1], *self._shape).to(torch.float32)
            std = (hi - lo) * torch.sigmoid(std) + lo
            dist_obj = td.Normal(torch.tanh(out), std)
            dist_obj = td.Independent(dist_obj, 1)
            dist_obj.minent = np.prod(self._shape) * td.Normal(0.99, lo).entropy()
            dist_obj.maxent = np.prod(self._shape) * td.Normal(0.0, hi).entropy()
            return dist_obj
        if self._dist == 'onehot':
            if self._unimix:
                probs = F.softmax(out, dim=-1)
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs = (1 - self._unimix) * probs + self._unimix * uniform
                dist_obj = self.get('OneHotDist', td.OneHotCategorical, probs=probs)
            else:
                dist_obj = self.get('OneHotDist', td.OneHotCategorical, logits=out)
            if len(self._shape) > 1:
                dist_obj = td.Independent(dist_obj, len(self._shape)-1)
            dist_obj.minent = 0.0
            dist_obj.maxent = np.prod(self._shape[:-1]) * np.log(self._shape[-1])
            return dist_obj
        raise NotImplementedError(self._dist)

# ---------------------------
# Conv2D
# ---------------------------
class Conv2D(Module):
    def __init__(self, depth, kernel, stride=1, transp=False, act='none', norm='none', pad='same', bias=True):
        super().__init__()
        self.transp = transp
        self.act = get_act(act)
        self.norm = Norm(norm)
        self.pad = pad
        self.stride = stride
        self.depth = depth
        self.bias = bias
        if transp:
            self.layer = nn.ConvTranspose2d(in_channels=0, out_channels=depth, kernel_size=kernel,
                                             stride=stride, bias=bias)
        else:
            self.layer = nn.Conv2d(in_channels=0, out_channels=depth, kernel_size=kernel,
                                   stride=stride, bias=bias)

    def forward(self, hidden):
        if hidden.dim() == 4:
            hidden = hidden.permute(0, 3, 1, 2)
        if self.layer.in_channels == 0:
            self.layer.in_channels = hidden.shape[1]
        hidden = self.layer(hidden)
        hidden = self.norm(hidden)
        hidden = self.act(hidden)
        hidden = hidden.permute(0, 2, 3, 1)
        return hidden

# ---------------------------
# Dense
# ---------------------------
class Dense(Module):
    def __init__(self, units, act='none', norm='none', bias=True):
        super().__init__()
        self._units = units
        self.act = get_act(act)
        self.norm_impl = norm
        self.bias = bias
        self.linear = None

    def forward(self, x):
        if self.linear is None:
            self.linear = self.get('linear', nn.Linear, x.shape[-1], self._units, bias=self.bias)
        x = self.linear(x)
        norm_layer = self.get('norm', Norm, self.norm_impl)
        x = norm_layer(x)
        x = self.act(x)
        return x

# ---------------------------
# Norm
# ---------------------------
class Norm(Module, nn.Module):
    def __init__(self, impl):
        super().__init__()
        self._impl = impl
        if impl == 'keras':
            self.layer = nn.LayerNorm(normalized_shape=0)
        elif impl == 'layer':
            self.scale = None
            self.offset = None

    def forward(self, x):
        if self._impl == 'none':
            return x
        elif self._impl == 'keras':
            if self.layer.normalized_shape == 0:
                self.layer.normalized_shape = x.shape[-1]
            return self.layer(x)
        elif self._impl == 'layer':
            if self.scale is None:
                self.scale = nn.Parameter(torch.ones(x.shape[-1], device=x.device))
                self.offset = nn.Parameter(torch.zeros(x.shape[-1], device=x.device))
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + 1e-3) * self.scale + self.offset
        else:
            raise NotImplementedError(self._impl)
