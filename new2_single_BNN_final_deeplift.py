from torch.nn import Module, Parameter
import xgcm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib import ticker
from cmcrameri import cm
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import intake
import xesmf as xe
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import cartopy
import numpy.polynomial.polynomial as poly
from netCDF4 import Dataset
from sklearn.cluster import KMeans
import cartopy.feature as cfeature
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import grad
import sys
import shap
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import io

from torch.nn import Module, Parameter
from captum.attr import IntegratedGradients, DeepLift, GradientShap, GuidedBackprop, LRP
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import grad
import sys
import shap
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

print('here')



def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out -1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


class BayesLinear(Module):
    r"""
    Applies Bayesian Linear
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()
    def reset_parameters(self):
        # Initialization method of BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

        # Initialization method of the original torch nn.linear.
#         init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
#         self.weight_log_sigma.data.fill_(self.prior_log_sigma)

#         if self.bias :
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias_mu, -bound, bound)

#             self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self) :
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps

        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else :
            bias = None

        return F.linear(input, weight, bias)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)


class BayesConv2d(nn.Module):
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(BayesConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.register_buffer('weight_eps', None)

        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self):
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias:
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self):
        self.weight_eps = None
        if self.bias:
            self.bias_eps = None

    def conv2d_forward(self, input, weight):
        if self.bias:
            if self.bias_eps is None:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else:
            bias = None

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return nn.functional.conv2d(nn.functional.pad(input, expanded_padding, mode='circular'),
                                        weight, bias, self.stride,
                                        padding=0, dilation=self.dilation, groups=self.groups)
        return nn.functional.conv2d(input, weight, bias, self.stride,
                                    padding=self.padding, dilation=self.dilation, groups=self.groups)

    def forward(self, input):
        if self.weight_eps is None:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps

        return self.conv2d_forward(input, weight)

    def extra_repr(self):
        return 'prior_mu={}, prior_sigma={}, in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={}, padding_mode={}'.format(
            self.prior_mu, self.prior_sigma, self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding, self.dilation, self.groups, self.bias, self.padding_mode)



class _BayesBatchNorm(Module):
    r"""
    Applies Bayesian Batch Normalization over a 2D or 3D input
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    """

    _version = 2
    __constants__ = ['prior_mu', 'prior_sigma', 'track_running_stats',
                     'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, prior_mu, prior_sigma, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_BayesBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.prior_mu = prior_mu
            self.prior_sigma = prior_sigma
            self.prior_log_sigma = math.log(prior_sigma)

            self.weight_mu = Parameter(torch.Tensor(num_features))
            self.weight_log_sigma = Parameter(torch.Tensor(num_features))
            self.register_buffer('weight_eps', None)

            self.bias_mu = Parameter(torch.Tensor(num_features))
            self.bias_log_sigma = Parameter(torch.Tensor(num_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('weight_mu', None)
            self.register_parameter('weight_log_sigma', None)
            self.register_buffer('weight_eps', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            # Initialization method of Adv-BNN.
            self.weight_mu.data.uniform_()
            self.weight_log_sigma.data.fill_(self.prior_log_sigma)
            self.bias_mu.data.zero_()
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

            # Initilization method of the original torch nn.batchnorm.
#             init.ones_(self.weight_mu)
#             self.weight_log_sigma.data.fill_(self.prior_log_sigma)
#             init.zeros_(self.bias_mu)
#             self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self) :
        if self.affine :
            self.weight_eps = torch.randn_like(self.weight_log_sigma)
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self) :
        if self.affine :
            self.weight_eps = None
            self.bias_eps = None

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.affine :
            if self.weight_eps is None :
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else :
            weight = None
            bias = None

        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{prior_mu}, {prior_sigma}, {num_features}, ' \
                'eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BayesBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class BayesBatchNorm2d(_BayesBatchNorm):
    r"""
    Applies Bayesian Batch Normalization over a 2D input
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following batchnorm of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class _BayesConvNd(Module):
    r"""
    Applies Bayesian Convolution
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    """
    __constants__ = ['prior_mu', 'prior_sigma', 'stride', 'padding', 'dilation',
                     'groups', 'bias', 'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_BayesConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        if transposed:
            self.weight_mu = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_log_sigma = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)
        else:
            self.weight_mu = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_log_sigma = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)

        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_log_sigma = Parameter(torch.Tensor(out_channels))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()


    def reset_parameters(self):
        # Initialization method of Adv-BNN.
        n = self.in_channels
        n *= self.kernel_size[0] ** 2
        stdv = 1.0 / math.sqrt(n)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)

        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self) :
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None

    def extra_repr(self):
        s = ('{prior_mu}, {prior_sigma}'
             ', {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is False:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_BayesConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction


def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.
    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).

    """
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()

def bayesian_kl_loss(model, reduction='mean', last_layer_only=False) :
    """
    An method for calculating KL divergence of whole layers in the model.
    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.

    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for m in model.modules() :
        if isinstance(m, (BayesLinear, BayesConv2d)):
            kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
            kl_sum += kl
            n += len(m.weight_mu.view(-1))

            if m.bias :
                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))

        if isinstance(m, BayesBatchNorm2d):
            if m.affine :
                kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))
    if last_layer_only or n == 0 :
        return kl

    if reduction == 'mean' :
        return kl_sum/n
    elif reduction == 'sum' :
        return kl_sum
    else :
        raise ValueError(reduction + " is not valid")

class BKLLoss(_Loss):
    """
    Loss for calculating KL divergence of baysian neural network model.
    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.
    """
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__(reduction)
        self.last_layer_only = last_layer_only

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)

class EarlyStopping:

    def __init__(self, patience=5,delta=0, verbose=False, path='checkpoint.pt',trace_func=print):

        #Args:
        """
                            Default: False
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.delta = delta
        self.val_loss_min = np.Inf
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def calculate_integrated_gradients_uniformbaseline(model, input_sequence, num_steps=100):
      # Generate a random baseline sequence
      mean = input_sequence.mean()  # Mean of the distribution
      stddev = input_sequence.std()  # Standard deviation of the distribution

      # Define the size of the tensor (similar to x_test)
      tensor_size = input_sequence.size()

      # Generate random samples from a Gaussian distribution
      baseline_sequence = torch.randn(tensor_size) * stddev + mean
      #baseline_sequence = torch.full(input_sequence.size(),constant)
      # Compute the difference sequence
      input_sequence=input_sequence.unsqueeze(0).expand(2,-1, -1, -1)
      baseline_sequence=baseline_sequence.unsqueeze(0).expand(2,-1, -1, -1)

      # Create the uniform baseline by sampling from the uniform distribution
      #baseline_sequence = torch.randn_like(input_sequence)

      # Compute the difference sequence
      diff_sequence = input_sequence - baseline_sequence

      # Calculate the step size for interpolation
      
      alpha = torch.linspace(0, 1, num_steps)

      # Initialize the integrated gradients
      integrated_gradients = torch.zeros_like(input_sequence)
      gradients_tensor = torch.zeros(num_steps, *input_sequence.shape)


      # Compute the integrated gradients
      for target_idx in range(2):
          for step in alpha:
             # Interpolate between the baseline and input sequences
             interpolated_sequence = baseline_sequence[target_idx,:, :,:] + step * diff_sequence[target_idx,:, :,:]

             # Enable gradients calculation
             interpolated_sequence.requires_grad_(True)

             # Forward pass through the model
             output = model(interpolated_sequence)

             # Backward pass to accumulate gradients
             grads = grad(output[target_idx,:].sum(), interpolated_sequence)[0]

             # Scale the gradients and accumulate
             integrated_gradients += grads * diff_sequence

      # Average the accumulated gradients
      integrated_gradients /= num_steps
      return integrated_gradients


def calculate_integrated_gradients_constbaseline(model, input_sequence, num_steps):
      # Generate a constant baseline sequence
      constant=0
      baseline_sequence = torch.full(input_sequence.size(),constant)
      # Compute the difference sequence
      input_sequence=input_sequence.unsqueeze(0).expand(2,-1, -1, -1)
      baseline_sequence=baseline_sequence.unsqueeze(0).expand(2,-1, -1, -1)
      diff_sequence = input_sequence - baseline_sequence

      # Calculate the step size for interpolation
      alpha = torch.linspace(0, 1, num_steps)

      # Initialize the integrated gradients
      integrated_gradients = torch.zeros_like(input_sequence)
      #integrated_gradients=integrated_gradients.unsqueeze(0).expand(2,-1, -1, -1)

      # Compute the integrated gradients

      for target_idx in range(2):
        for step in alpha:
          # Interpolate between the baseline and input sequences
          interpolated_sequence = baseline_sequence[target_idx,:, :,:] + step * diff_sequence[target_idx,:, :,:]

          # Enable gradients calculation
          interpolated_sequence.requires_grad_(True)

          # Forward pass through the model
          output = model(interpolated_sequence)

          # Backward pass to accumulate gradients
          grads = grad(output[target_idx,:].sum(), interpolated_sequence,allow_unused=True)[0]

          # Scale the gradients and accumulate
          integrated_gradients += grads * diff_sequence

        # Average the accumulated gradients
      integrated_gradients /= num_steps
      return integrated_gradients

import copy

def ts_diff(ts):
    diff_ts = [0] * len(ts)
    for i in range(1, len(ts)):
        diff_ts[i] = ts[i] - ts[i - 1]
    return diff_ts

#def ts_int(ts_diff, ts_base, start=0):
#    ts_diff = np.asarray(ts_diff)  # Convert to NumPy array for vectorized operations
#    ts_base = np.asarray(ts_base)

#    ts = np.empty_like(ts_diff)  # Create an empty array to store the integrated time series
#    ts[0] = start + ts_diff[0]  # Set the initial value

    # Perform vectorized addition to calculate the integrated series
#    ts[1:] = ts_diff[1:] + ts_base[:-1]

#    return ts.tolist()
def ts_int(ts_diff, ts_base, start=0):
        ts_diff = np.asarray(ts_diff)  # Convert to NumPy array for vectorized operations
        ts_base = np.asarray(ts_base)

        ts = np.empty_like(ts_diff)  # Create an empty array to store the integrated time series
        ts[0] = start + ts_diff[0]  # Set the initial value

        # Perform vectorized addition to calculate the integrated series
        ts[1:] = ts_diff[1:] + ts_base[:-1]

        return ts.tolist()


def sliding_window(ts, features, target_len ):
    X = []
    Y = []

    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len):i - target_len])
        Y.append(ts[i - target_len:i])

    return np.array(X),np.array(Y)


class ModelWrapper(nn.Module):
    def __init__(self, model, target_dim):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.target_dim = target_dim

    def forward(self, x):
        return self.model(x)[:, :, self.target_dim]

def lrp(model, x, baseline,epsilon):
    # Forward pass
    x=x.unsqueeze(0).expand(2,-1, -1, -1)
    baseline=baseline.unsqueeze(0).expand(2,-1, -1, -1)

    for target_idx in range(2):
       # Initialize relevance scores
       diff_sequence = x[target_idx,:,:,:] - baseline[target_idx,:,:,:]
       x.requires_grad = True
       output = model(x)
       R= output[target_idx,:].clone()

       # Backward pass with LRP
       output[target_idx,:].backward(gradient=R, retain_graph=True)
       # Compute feature importances
       relevance_scores = x.grad * diff_sequence
       relevance_scores = relevance_scores/(x.grad.sum(dim=2, keepdim=True) + epsilon)
       # Clear gradients for future computations
       model.zero_grad()

    return relevance_scores


def lrp(model, x, baseline, alpha, beta, epsilon):
    # Forward pass
    x = x.unsqueeze(0).expand(1, -1, -1, -1)
    baseline = baseline.unsqueeze(0).expand(1, -1, -1, -1)

    for target_idx in range(1):
        # Initialize relevance scores
        diff_sequence = x[target_idx, :, :, :] - baseline[target_idx, :, :, :]
        x.requires_grad = True
        output = model(x)
        R = output[target_idx, :].clone()

        # Backward pass with LRP-alpha-beta
        output[target_idx, :].backward(gradient=R, retain_graph=True)
        
        # Compute feature importances
        relevance_scores = x.grad * diff_sequence
        relevance_scores = relevance_scores / (x.grad.sum(dim=2, keepdim=True) + epsilon)

        # Apply alpha-beta parameters
        pos_contrib = relevance_scores.clamp(min=0) ** alpha
        neg_contrib = relevance_scores.clamp(max=0) ** beta
        relevance_scores = pos_contrib - neg_contrib

        # Clear gradients for future computations
        model.zero_grad()

    return relevance_scores



C=[1,24,48,60,72,96,120]

members=[10]
prediction1=np.zeros((264,members[0]))
prediction2=np.zeros((264,members[0]))
prediction3=np.zeros((264,members[0]))
prediction4=np.zeros((264,members[0]))
prediction5=np.zeros((264,members[0]))
prediction6=np.zeros((264,members[0]))
prediction7=np.zeros((264,members[0]))
prediction8=np.zeros((264,members[0]))
prediction9=np.zeros((264,members[0]))


skill=np.zeros((9))

feature=[1]

Z0_DL_24_1=np.zeros((264,members[0]))
Z1_DL_24_1=np.zeros((264,members[0]))
Z2_DL_24_1=np.zeros((264,members[0]))
Z3_DL_24_1=np.zeros((264,members[0]))
Z4_DL_24_1=np.zeros((264,members[0]))
Z5_DL_24_1=np.zeros((264,members[0]))
Z6_DL_24_1=np.zeros((264,members[0]))

Z0_DL_96_1=np.zeros((264,members[0]))
Z1_DL_96_1=np.zeros((264,members[0]))
Z2_DL_96_1=np.zeros((264,members[0]))
Z3_DL_96_1=np.zeros((264,members[0]))
Z4_DL_96_1=np.zeros((264,members[0]))
Z5_DL_96_1=np.zeros((264,members[0]))
Z6_DL_96_1=np.zeros((264,members[0]))





A_lrp0_24=np.zeros((feature[0],264,members[0]))
A_lrp1_24=np.zeros((feature[0],264,members[0]))
A_lrp2_24=np.zeros((feature[0],264,members[0]))
A_lrp3_24=np.zeros((feature[0],264,members[0]))
A_lrp4_24=np.zeros((feature[0],264,members[0]))
A_lrp5_24=np.zeros((feature[0],264,members[0]))
A_lrp6_24=np.zeros((feature[0],264,members[0]))
A_lrp7_24=np.zeros((feature[0],264,members[0]))

A_lrp0_96=np.zeros((feature[0],264,members[0]))
A_lrp1_96=np.zeros((feature[0],264,members[0]))
A_lrp2_96=np.zeros((feature[0],264,members[0]))
A_lrp3_96=np.zeros((feature[0],264,members[0]))
A_lrp4_96=np.zeros((feature[0],264,members[0]))
A_lrp5_96=np.zeros((feature[0],264,members[0]))
A_lrp6_96=np.zeros((feature[0],264,members[0]))
A_lrp7_96=np.zeros((feature[0],264,members[0]))


w=0
M=[12,24,36,48,60,72,84,96,108,120,132,144,156,168,180,192,204,216,228,240,252,264]
A_mean_baseline1_24_1=np.zeros((264,members[0],4))
A_mean_baseline1_96_1=np.zeros((264,members[0],4))
A_mean_baseline1_24_2=np.zeros((264,members[0],4))
A_mean_baseline1_96_2=np.zeros((264,members[0],4))
A_mean_baseline1_24_3=np.zeros((264,members[0],4))
A_mean_baseline1_96_3=np.zeros((264,members[0],4))

print('here2')

import random
seed2=[0,10,20,30,40,50,60,70,80,90]


# Ensure deterministic behavior in PyTorch
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


U1=['data_NADW.mat']
skill=np.zeros((9,10))

lags = range(97)
persistence=np.zeros((3,97))
import statsmodels.api as sm
for w in range(1):
    w=0
    data = io.loadmat(U1[w])
    # Extract variables X and 
    T = data['T']
    acorr = sm.tsa.acf(T.reshape(1980,), nlags = len(lags)-1)
    #auto2[im,:]=acorr
    #acorr = sm.tsa.acf(savitzky_golay(A2[:,im],12,3), nlags = len(lags)-1)
    persistence[w,:]=acorr
    

    plt.figure()
    plt.plot(persistence[w,::12])
    plt.ylabel('NADW oxygen persistence',fontsize=13)
    plt.xlabel('time',fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig('o2_'+str(w)+'_persistence.pdf')

    
#    ts = np.empty_like(ts_diff)  # Create an empty array to store the integrated time series
#    ts[0] = start + ts_diff[0]  # Set the initial value

    # Perform vectorized addition to calculate the integrated series
#    ts[1:] = ts_diff[1:] + ts_base[:-1]
#    return ts.tolist() 


def calculate_deeplift_constbaseline(model, input_sequence, targets):
    dl = DeepLift(model)
    constant =0

    #baseline_sequence = torch.full(input_sequence.size(), constant, device=input_sequence.device)
    baseline_sequence = torch.normal(mean=0.0, std=1.0, size=input_sequence.size(), device=input_sequence.device)
    attributions = []
    for target in targets:
        attr = dl.attribute(input_sequence, baselines=baseline_sequence, target=target)
        attributions.append(attr.cpu().detach().numpy())
    return np.array(attributions)

skill=np.zeros((9,10))
IG_NADW1=np.zeros((4,264,10))
IG_LCDW1=np.zeros((4,264,10))
IG_UCDW1=np.zeros((4,264,10))

IG_NADW2=np.zeros((4,264,10))
IG_LCDW2=np.zeros((4,264,10))
IG_UCDW2=np.zeros((4,264,10))
ig_attributions_NADW=[]
ig_attributions_UCDW=[]
ig_attributions_LCDW=[]

for r in range(10):
 ig_attributions_NADW=[]
 ig_attributions_UCDW=[]
 ig_attributions_LCDW=[]
 ig_attributions_all1=[]
 for w in range(1):
    w=0
    if w ==0:
        seed = seed2[r]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mse_loss = nn.MSELoss()
        kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.1
        #    optimizer = optim.Adam(model.parameters(), lr=0.0005)
        input2=[96]
        output2=[96]
        test_len=input2[0]+output2[0]
        #torch.cuda.manual_seed_all(seed)
        
        data = io.loadmat('data_NADW.mat')
        # Extract variables X and y

        thetao_index_NADW = data['thetao_index_NADW']
        thetao_index_NADW=thetao_index_NADW.reshape(1980,1)
        so_index_NADW = data['so_index_NADW']
        so_index_NADW=so_index_NADW.reshape(1980,1)
        eke_index_NADW = data['eke_index_NADW']
        eke_index_NADW=eke_index_NADW.reshape(1980,1)
        N_index_NADW = data['N_index_NADW']
        N_index_NADW=N_index_NADW.reshape(1980,1)

        T = data['T']
        D=np.concatenate((thetao_index_NADW,so_index_NADW,eke_index_NADW,N_index_NADW),axis=1)
        T=T.reshape(1980,1)
        D2=np.zeros((1980,4))
        for j in range(D.shape[1]):
            D2[:,j]=ts_diff(D[:,j])
        T2=np.zeros((1980,1))
        for j in range(T.shape[1]):
            T2[:,j]=T[:,j]
        X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])

        #T2=np.zeros((1980,1))
        #for j in range(T.shape[1]):
        #    T2[:,j]=ts_diff(T[:,j])    
        
        test_len=96+96
        #T=np.concatenate((o2_index[:,np.newaxis],zos_index[:,np.newaxis],so_index[:,np.newaxis],npp_index[:,np.newaxis]),axis=1)

        train_ratio=0.8
        train_len = round(len(X_ss[:(-test_len-264)]) * train_ratio)
        test_len=input2[0]+output2[0] #150/3
        X_train, Y_train= X_ss[:(-test_len-264)],\
                                       Y_mm[:(-test_len-264)]
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]


        print("X_train",X_train.shape)
        X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]

        x_train = torch.tensor(data = X_train).float()
        y_train = torch.tensor(data = Y_train).float()

        x_val = torch.tensor(data = X_val).float()
        y_val = torch.tensor(data = Y_val).float()

        class ModelWithAttention(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_heads, output_dim):
               super(ModelWithAttention, self).__init__()
               self.fc1 = BayesLinear(prior_mu=0, prior_sigma=1, in_features=input_dim, out_features=hidden_dim)
               self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
               self.fc2 = BayesLinear(prior_mu=0, prior_sigma=1, in_features=hidden_dim, out_features=output_dim)

            def forward(self, x):
               x = F.relu(self.fc1(x))
               x, _ = self.attention(x, x, x)  # Apply Multihead Attention
               x = self.fc2(x)
               return x

        input_dim = 4
        hidden_dim = 6
        num_heads = 3
        output_dim = 1
        model = ModelWithAttention(input_dim, hidden_dim, num_heads, output_dim)


        training_loss = []
        validation_loss = []

    
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        mse_loss = nn.MSELoss()
        kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
    
    best_params = None
    min_avg_val_loss = float('inf')
    best_kl_weight = None
    kl_weights = [0.01, 0.1, 0.9]  # List of kl_weight values to try
    kf = KFold(n_splits=5, shuffle=True, random_state=21)  # Set up K-Fold cross-validation


    for kl_weight in kl_weights:
        fold_val_losses = []  # Store validation loss for each fold

        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
            print(f"Fold {fold + 1}/{kf.get_n_splits()}")

            # Split data into train and validation sets for this fold
            x_fold_train, x_fold_val = x_train[train_idx], x_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]


            optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            early_stopping = EarlyStopping(patience=70, verbose=True)

            fold_min_val_loss = float('inf')  # Track the best val loss for this fold

            for step in range(1500):
                model.train()
                optimizer.zero_grad()

                # Forward pass
                pre = model(x_fold_train)
                mse = mse_loss(pre, y_fold_train)
                kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
                kl = kl_loss(model)
                loss = mse + kl_weight * kl
                loss.backward()

                optimizer.step()
                #scheduler.step()

                # Validation within the training step
                model.eval()
                with torch.no_grad():
                    pre_val = model(x_fold_val)
                    val_loss = mse_loss(pre_val, y_fold_val)
                    fold_val_losses.append(val_loss.item())

                    # Save best model parameters for this fold
                    if val_loss.item() < fold_min_val_loss:
                       fold_min_val_loss = val_loss.item()
                       best_params = model.state_dict().copy()

                    # Early stopping
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                       print(f"Early stopping in fold {fold + 1} at step:", step)
                       break

                # Calculate average validation loss for this kl_weight across all folds
                avg_val_loss = np.mean(fold_val_losses)

                # Check if this kl_weight gives the best average validation loss
                if avg_val_loss < min_avg_val_loss:
                   min_avg_val_loss = avg_val_loss
                   best_kl_weight = kl_weight
                   best_params = model.state_dict().copy()

    print(f"KL weight: {kl_weight}, Avg Validation Loss: {avg_val_loss:.6f}")

    # After K-Fold CV, load the best model parameters
    print(f"Best KL weight: {best_kl_weight}, with Avg Validation Loss: {min_avg_val_loss:.6f}")
    model.load_state_dict(best_params)


      





    for N in range(264):
       if N==0:
         if w ==0:
            
            X_test, Y_test= X_ss[-96:],Y_mm[-96:]
            x_test = torch.tensor(data = X_test).float()
            y_test = torch.tensor(data = Y_test).float()

            y_predict = model(x_test)

            Z=ts_int(
            y_predict[-1,:,0].tolist(),
            T[-96:,0],
            start = T[-96-1,0]
            )
            #y_test=y_test[:,:,0] 

            prediction1[0,r]=Z[-1]
            prediction2[0,r]=Z[-12]
            prediction3[0,r]=Z[-24]
            prediction4[0,r]=Z[-36]
            prediction5[0,r]=Z[-48]
            prediction6[0,r]=Z[-60]
            prediction7[0,r]=Z[-72]
            prediction8[0,r]=Z[-84]
            
            ig = DeepLift(model)
            #baseline_sequence = torch.full(x_test.size(), constant, device=x_test.device)
            constant=1
            num_steps=200
            #baseline_sequence = torch.normal(mean=0.0, std=1.0, size=x_test.size(), device=x_test.device)
            #baseline_sequence = torch.full(x_test.size(), constant, device=x_test.device)
            targets = list(range(1))
            model_ts1 = ModelWrapper(model, target_dim=0)
            #ig_ts1 = calculate_integrated_gradients_constbaseline(model_ts1, x_test, targets=targets)
            dl_ts1 = calculate_deeplift_constbaseline(model_ts1, x_test, targets=targets)
            ig_attributions_all1.append(dl_ts1)
 
       if N>0:
         if w ==0:
            
            X_test, Y_test= X_ss[-96-N:-N],Y_mm[-96-N:-N]
            x_test = torch.tensor(data = X_test).float()
            y_test = torch.tensor(data = Y_test).float()

            y_predict = model(x_test)

            Z=ts_int(
            y_predict[-1,:,0].tolist(),
            T[-96-N:-N,0],
            start = T[-96-N-1,0]
            )
            #y_test=y_test[:,:,0] 

            prediction1[N,r]=Z[-1]
            prediction2[N,r]=Z[-12]
            prediction3[N,r]=Z[-24]
            prediction4[N,r]=Z[-36]
            prediction5[N,r]=Z[-48]
            prediction6[N,r]=Z[-60]
            prediction7[N,r]=Z[-72]
            prediction8[N,r]=Z[-84]
            
            A=np.corrcoef(T[-264:,0],prediction1[::-1,r])
            skill[8,r]=A[1][0]
            A=np.corrcoef(T[-264-12:-12,0],prediction2[::-1,r])
            skill[7,r]=A[1][0]
            A=np.corrcoef(T[-264-24:-24,0],prediction3[::-1,r])
            skill[6,r]=A[1][0]
            A=np.corrcoef(T[-264-36:-36,0],prediction4[::-1,r])
            skill[5,r]=A[1][0]
            A=np.corrcoef(T[-264-48:-48,0],prediction5[::-1,r])
            skill[4,r]=A[1][0]
            A=np.corrcoef(T[-264-60:-60,0],prediction6[::-1,r])
            skill[3,r]=A[1][0]
            A=np.corrcoef(T[-264-72:-72,0],prediction7[::-1,r])
            skill[2,r]=A[1][0]
            A=np.corrcoef(T[-264-84:-84,0],prediction8[::-1,r])
            skill[1,r]=A[1][0]
            skill[0,r]=1
            
            ig = DeepLift(model)
            #baseline_sequence = torch.full(x_test.size(), constant, device=x_test.device)
            constant=1
            num_steps=200
            #baseline_sequence = torch.normal(mean=0.0, std=1.0, size=x_test.size(), device=x_test.device) 
            #baseline_sequence = torch.full(x_test.size(), constant, device=x_test.device)
            targets = list(range(1))
            model_ts1 = ModelWrapper(model, target_dim=0)
            #ig_ts1 = calculate_integrated_gradients_constbaseline(model_ts1, x_test, targets=targets)
            dl_ts1 = calculate_deeplift_constbaseline(model_ts1, x_test, targets=targets)
            ig_attributions_all1.append(dl_ts1)
    
    ig_attributions_all1=np.array(ig_attributions_all1)
    ig_attributions_all1= ig_attributions_all1.reshape(264,96,96,4)
    Z0_DL_24_1[:,r]= ig_attributions_all1[:,-1,-72,0]
    Z1_DL_24_1[:,r]= ig_attributions_all1[:,-1,-72,1]
    Z2_DL_24_1[:,r]= ig_attributions_all1[:,-1,-72,2]
    Z3_DL_24_1[:,r]= ig_attributions_all1[:,-1,-72,3]

    Z0_DL_96_1[:,r]= ig_attributions_all1[:,-1,-1,0]
    Z1_DL_96_1[:,r]= ig_attributions_all1[:,-1,-1,1]
    Z2_DL_96_1[:,r]= ig_attributions_all1[:,-1,-1,2]
    Z3_DL_96_1[:,r]= ig_attributions_all1[:,-1,-1,3]         
    
    A_mean_baseline1_24_1[:,r,:]=np.concatenate((Z0_DL_24_1[:,r,np.newaxis],Z1_DL_24_1[:,r,np.newaxis],Z2_DL_24_1[:,r,np.newaxis],Z3_DL_24_1[:,r,np.newaxis]),axis=1)
    A_mean_baseline1_96_1[:,r,:]=np.concatenate((Z0_DL_96_1[:,r,np.newaxis],Z1_DL_96_1[:,r,np.newaxis],Z2_DL_96_1[:,r,np.newaxis],Z3_DL_96_1[:,r,np.newaxis]),axis=1)
    

years=np.arange(1850,2015,1/12)
plt.figure()
for i in range(10):
    w=0
    #plt.fill_between(years[-264:],(prediction1[::-1]-ci), (prediction1[::-1]+ci), color='green', alpha=0.5)
    plt.plot(years[-264:],prediction1[::-1,i],color='g',linewidth=0.5)
    plt.plot(years[-264:],np.mean(prediction1[::-1,:],axis=1),color='k',linewidth=1.75)
    plt.plot(years[-264:],T[-264:,],color='r',linewidth=1.75)
    plt.ylabel("O2 anomaly",fontsize=13)
    plt.xlabel("Years",fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

plt.savefig('prediction_8_NADW_'+str(w)+'.pdf')

plt.figure()
for i in range(10):
    w=0
    #plt.fill_between(years[-264:],(prediction1[::-1]-ci), (prediction1[::-1]+ci), color='green', alpha=0.5)
    plt.plot(years[-264-72:-72],prediction7[::-1,i],color='g',linewidth=0.5)
    plt.plot(years[-264-72:-72],np.mean(prediction7[::-1,:],axis=1),color='k',linewidth=1.75)
    plt.plot(years[-264-72:-72],T[-264-72:-72,0],color='r',linewidth=1.75)
    plt.ylabel("O2 anomaly",fontsize=13)
    plt.xlabel("Years",fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

plt.savefig('prediction_2_NADW_'+str(w)+'.pdf')


print('skill',skill)


years=np.arange(1850,2015,1/12)




def calculate_entropy(logits):
      # Calculate probabilities using logits
      exp_logits = np.exp(logits)
      print("exp_logits",exp_logits.shape)
      probabilities = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
      print("probabilities",probabilities)
      # Calculate entropy
      entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8), axis=1)
      return entropy

def calculate_joint_entropy(*logits):
      # Calculate probabilities using logits
      exp_logits = [np.exp(logit) for logit in logits]
      probabilities = [exp_logit / np.sum(exp_logit, axis=0, keepdims=True) for exp_logit in exp_logits]

      # Calculate joint entropy
      joint_entropy = -np.sum(np.prod(probabilities, axis=0) * np.log2(np.prod(probabilities, axis=0) + 1e-8), axis=1)

      return joint_entropy

feature = ['PT','Salinity','EKE','N']

new_x_ticks = [0,1, 2, 3]  # Your new x-axis tick values
new_x_labels = ['PT','Salinity','EKE','N']


# u undo the canceling
#print("shap",np.mean(A_mean_baseline1_24,axis=0))
#print('ig',A_mean_baseline1_24[:Q,:,:,:])
#print('ig',np.nanmean(np.nanmean(np.nanmean(A_mean_baseline1_24[:Q,:,:,:],axis=0),axis=0),axis=1).shape)
stds_24_1 = (np.nanstd(np.nanstd(A_mean_baseline1_24_1[:,:,:], axis=0),axis=0).reshape(4,))**(2)
stds_96_1 = (np.nanstd(np.nanstd(A_mean_baseline1_96_1[:,:,:], axis=0),axis=0).reshape(4,))**(2)


print('x_train',x_train.shape)
print('x_test',x_test.shape)
#print('ig',A_mean_baseline1_24[:Q,:,:,:])


#### entropy
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

def shannon_entropy(time_series):
    # Calculate the probability distribution for unique values
    unique_values, value_counts = np.unique(time_series, return_counts=True)

    # Calculate the logits (log-odds) for each unique value
    logits = value_counts - np.max(value_counts)  # Subtract the max value for numerical stability

    # Calculate softmax probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits))

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-5))  # Adding a small epsilon to prevent log(0)

    return entropy


# Example usage


import numpy as np

def shannon_entropy(sequence, base=2):
    # Create histogram with bins
    histogram, bin_edges = np.histogram(sequence, bins='auto')

    # Calculate probabilities of values falling into each bin
    probabilities = histogram / len(sequence)

    # Remove zero probabilities to avoid NaN in the entropy calculation
    probabilities = probabilities[probabilities > 0]

    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log(probabilities) / np.log(base))

    return entropy

# Calculate Shannon entropy for each time series in time_series_list
entropy_values=[]





from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
forecast_length = [96]
accuracy = np.zeros(9)  # Initialize an array to store accuracy values

predictions=np.zeros((264,5))
predictions1=np.zeros((264,5))
predictions2=np.zeros((264,5))
predictions3=np.zeros((264,5))
predictions4=np.zeros((264,5))
predictions5=np.zeros((264,5))
predictions6=np.zeros((264,5))
predictions7=np.zeros((264,5))
predictions8=np.zeros((264,5))
predictions9=np.zeros((264,5))



from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import datasets, linear_model

forecast_length = 96
accuracy=np.zeros((9,1))
accuracy_lasso=np.zeros((9,1))



for w in range(1):
  w=0 
  
  data = io.loadmat('data_NADW.mat')
  # Extract variables X and y

  thetao_index_NADW = data['thetao_index_NADW']
  thetao_index_NADW=thetao_index_NADW.reshape(1980,1)
  so_index_NADW = data['so_index_NADW']
  so_index_NADW=so_index_NADW.reshape(1980,1)
  eke_index_NADW = data['eke_index_NADW']
  eke_index_NADW=eke_index_NADW.reshape(1980,1)
  N_index_NADW = data['N_index_NADW']
  N_index_NADW=N_index_NADW.reshape(1980,1)

  T = data['T']
  D=np.concatenate((thetao_index_NADW,so_index_NADW,eke_index_NADW,N_index_NADW),axis=1)
  D2=np.zeros((1980,4))
  for j in range(D.shape[1]):
      D2[:,j]=ts_diff(D[:,j])
  #T2=np.zeros((1980,1))
  T=T.reshape(1980,)
  T2=np.zeros((1980,))
  for j in range(1):
      T2[:,]=T[:,]
  X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])

  T=T.reshape(1980,)

  print("X_ss",X_ss.shape)
  print("y_mm",Y_mm.shape)
  train_ratio=0.7
  train_len = round(len(X_ss[:-(96+96+264)]) * train_ratio)
  test_len=186 #150/3


  threshold = 0.5  # Adjust this threshold as needed for binary classification
  #X_train, y_train = create_binary_sequences(train_data, input_length, output_length)
  #X_test, y_test = create_binary_sequences(test_data, input_length, output_length)
  X_train,y_train=X_ss[:-(96+96+264)],Y_mm[:-(96+96+264)]
  X_train = X_train.reshape(X_train.shape[0], -1)

  X_test,y_test=X_ss[-(96+96+264):],Y_mm[-(96+96+264):]
  X_test = X_test.reshape(X_test.shape[0], -1)

  # Create and fit a logistic regression model

  model = linear_model.LinearRegression()#LinearRegression()
  alpha = 0.5  # Regularization strength (adjus
  model.fit(X_train, y_train)
  # Make predictions on the test data


  for N in range(264):
    if N == 0:
        X_test, Y_test = X_ss[-96:], Y_mm[-96:]
    
        X_test = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test)

        Z=ts_int(
                y_pred[-1,:].tolist(),
            T[-96:,],
            start = T[-96 - 1,]
            )
        predictions1[N,w] = Z[-1]
        predictions2[N,w] = Z[-12]
        predictions3[N,w] = Z[-24]
        predictions4[N,w] = Z[-36]
        predictions5[N,w] = Z[-48]
        predictions6[N,w] = Z[-60]
        predictions7[N,w] = Z[-72]
        predictions8[N,w] = Z[-84]
        
    if N>0:
        X_test, Y_test = X_ss[-96-N:-N], Y_mm[-96-N:-N]

        X_test = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test)

        Z=ts_int(
            y_pred[-1,:].tolist(),
            T[-96-N:-N,],
            start = T[-96-N - 1,]
            )
        predictions1[N,w] = Z[-1]
        predictions2[N,w] = Z[-12]
        predictions3[N,w] = Z[-24]
        predictions4[N,w] = Z[-36]
        predictions5[N,w] = Z[-48]
        predictions6[N,w] = Z[-60]
        predictions7[N,w] = Z[-72]
        predictions8[N,w] = Z[-84]
        

#real_hist, _ = np.histogram(o2_index[-264:,], bins=10, density=True)
# Calculate the correlation coefficient between test and predictions

  accuracy[8,w] = np.corrcoef(T[-264:,], predictions1[::-1,w])[1][0]
  accuracy[7,w] = np.corrcoef(T[-264-12:-12,], predictions2[::-1,w])[1][0]
  accuracy[6,w] = np.corrcoef(T[-264-24:-24,], predictions3[::-1,w])[1][0]
  accuracy[5,w] = np.corrcoef(T[-264-36:-36,], predictions4[::-1,w])[1][0]
  accuracy[4,w] = np.corrcoef(T[-264-48:-48,], predictions5[::-1,w])[1][0] 
  accuracy[3,w] = np.corrcoef(T[-264-60:-60,], predictions6[::-1,w])[1][0]
  accuracy[2,w] = np.corrcoef(T[-264-72:-72,], predictions7[::-1,w])[1][0]
  accuracy[1,w] = np.corrcoef(T[-264-84:-84,], predictions8[::-1,w])[1][0]
  accuracy[0,w] = 1

  plt.figure()
  plt.plot(accuracy[:,w])
  plt.xlabel("tleads",fontsize=13)
  plt.ylabel("skill",fontsize=13)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.savefig('skill_linearreg_'+str(w)+'.png')


#print('liner',accuracy[:,0])
#print('liner',accuracy[:,1])

A_lasso_24=np.zeros((264,4))
A_lasso_96=np.zeros((264,4))

coeff0_96=np.zeros((264))
coeff1_96=np.zeros((264))
coeff2_96=np.zeros((264))
coeff3_96=np.zeros((264))
coeff4_96=np.zeros((264))
coeff5_96=np.zeros((264))
coeff6_96=np.zeros((264))

coeff0_24=np.zeros((264))
coeff1_24=np.zeros((264))
coeff2_24=np.zeros((264))
coeff3_24=np.zeros((264))
coeff4_24=np.zeros((264))
coeff5_24=np.zeros((264))
coeff6_24=np.zeros((264))

predictions1=np.zeros((264,5))
predictions2=np.zeros((264,5))
predictions3=np.zeros((264,5))
predictions4=np.zeros((264,5))
predictions5=np.zeros((264,5))
predictions6=np.zeros((264,5))
predictions7=np.zeros((264,5))
predictions8=np.zeros((264,5))
predictions9=np.zeros((264,5))


for w in range(1):
  w=0 
  data = io.loadmat('data_NADW.mat')
  # Extract variables X and y

  thetao_index_NADW = data['thetao_index_NADW']
  thetao_index_NADW=thetao_index_NADW.reshape(1980,1)
  so_index_NADW = data['so_index_NADW']
  so_index_NADW=so_index_NADW.reshape(1980,1)
  eke_index_NADW = data['eke_index_NADW']
  eke_index_NADW=eke_index_NADW.reshape(1980,1)
  N_index_NADW = data['N_index_NADW']
  N_index_NADW=N_index_NADW.reshape(1980,1)

  T = data['T']
  D=np.concatenate((thetao_index_NADW,so_index_NADW,eke_index_NADW,N_index_NADW),axis=1)
  D2=np.zeros((1980,4))
  for j in range(D.shape[1]):
      D2[:,j]=ts_diff(D[:,j])
  #T2=np.zeros((1980,1))
  T=T.reshape(1980,)
  T2=np.zeros((1980,))
  for j in range(1):
      T2[:,]=T[:,]
  T=T.reshape(1980,)
  #T2=np.zeros((1980,))  
  #T2[:,]=ts_diff(T[:,])

  X_ss, Y_mm =  split_sequences(D2,T,96,96)
  print("X_ss",X_ss.shape)
  print("y_mm",Y_mm.shape)
  train_ratio=0.7
  train_len = round(len(X_ss[:-(96+96+264)]) * train_ratio)
  test_len=216 #150/3


  threshold = 0.5  # Adjust this threshold as needed for binary classification
  #X_train, y_train = create_binary_sequences(train_data, input_length, output_length)
  #X_test, y_test = create_binary_sequences(test_data, input_length, output_length)
  X_train,y_train=X_ss[:-(96+96+264)],Y_mm[:-(96+96+264)]
  X_train = X_train.reshape(X_train.shape[0], -1)

  X_test,y_test=X_ss[-(96+96+264):],Y_mm[-(96+96+264):]
  X_test = X_test.reshape(X_test.shape[0], -1)

  # Create and fit a logistic regression model

  model = linear_model.Lasso(alpha=0.00008)  # 0.0006 Adjust alpha as needed for regularization strength
  model.fit(X_train, y_train)
  model_coef = np.abs(model.coef_)
  # Make predictions on the test data
  
  for N in range(264):
    if N == 0:
        X_test, Y_test = X_ss[-96:], Y_mm[-96:]

        X_test = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test)

        Z=ts_int(
            y_pred[:,-1].tolist(),
            T[-96:,],
            start = T[-96 - 1,]
            )
        predictions1[N,w] = Z[-1]
        predictions2[N,w] = Z[-12]
        predictions3[N,w] = Z[-24]
        predictions4[N,w] = Z[-36]
        predictions5[N,w] = Z[-48]
        predictions6[N,w] = Z[-60]
        predictions7[N,w] = Z[-72]
        predictions8[N,w] = Z[-84]
        
        A=model_coef.reshape(96,96,4)
        A=np.nanmean(A,axis=0)

        coeff0_96[N]=A[-1,0]
        coeff1_96[N]=A[-1,1]
        coeff2_96[N]=A[-1,2]
        coeff3_96[N]=A[-1,3]
        
        coeff0_24[N]=A[-84,0]
        coeff1_24[N]=A[-84,1]
        coeff2_24[N]=A[-84,2]
        coeff3_24[N]=A[-84,3]
        

        A_lasso_24[N,:]=np.concatenate((coeff0_24[N,np.newaxis],coeff1_24[N,np.newaxis],coeff2_24[N,np.newaxis],coeff3_24[N,np.newaxis]),axis=0)
        A_lasso_96[N,:]=np.concatenate((coeff0_96[N,np.newaxis],coeff1_96[N,np.newaxis],coeff2_96[N,np.newaxis],coeff3_96[N,np.newaxis]),axis=0)
    if N>0:
        X_test, Y_test = X_ss[-96-N:-N], Y_mm[-96-N:-N]

        X_test = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test)

        Z=ts_int(
                y_pred[-1,:].tolist(),
            T[-96-N:-N,],
            start = T[-96-N - 1,]
            )
        predictions1[N,w] = Z[-1]
        predictions2[N,w] = Z[-12]
        predictions3[N,w] = Z[-24]
        predictions4[N,w] = Z[-36]
        predictions5[N,w] = Z[-48]
        predictions6[N,w] = Z[-60]
        predictions7[N,w] = Z[-72]
        predictions8[N,w] = Z[-84]
        
        A=model_coef.reshape(96,96,4)
        A=np.nanmean(A,axis=0)

        coeff0_96[N]=A[-1,0]
        coeff1_96[N]=A[-1,1]
        coeff2_96[N]=A[-1,2]
        coeff3_96[N]=A[-1,3]
    

        coeff0_24[N]=A[-84,0]
        coeff1_24[N]=A[-84,1]
        coeff2_24[N]=A[-84,2]
        coeff3_24[N]=A[-84,3]
            
  
        A_lasso_24[N,:]=np.concatenate((coeff0_24[N,np.newaxis],coeff1_24[N,np.newaxis],coeff2_24[N,np.newaxis],coeff3_24[N,np.newaxis]),axis=0)
        A_lasso_96[N,:]=np.concatenate((coeff0_96[N,np.newaxis],coeff1_96[N,np.newaxis],coeff2_96[N,np.newaxis],coeff3_96[N,np.newaxis]),axis=0)
  accuracy_lasso[8,w] = np.corrcoef(T[-264:,], predictions1[::-1,w])[1][0]
  accuracy_lasso[7,w] = np.corrcoef(T[-264-12:-12,], predictions2[::-1,w])[1][0]
  accuracy_lasso[6,w] = np.corrcoef(T[-264-24:-24,], predictions3[::-1,w])[1][0]
  accuracy_lasso[5,w] = np.corrcoef(T[-264-36:-36,], predictions4[::-1,w])[1][0]
  accuracy_lasso[4,w] = np.corrcoef(T[-264-48:-48,], predictions5[::-1,w])[1][0]
  accuracy_lasso[3,w] = np.corrcoef(T[-264-60:-60,], predictions6[::-1,w])[1][0]
  accuracy_lasso[2,w] = np.corrcoef(T[-264-72:-72,], predictions7[::-1,w])[1][0]
  accuracy_lasso[1,w] = np.corrcoef(T[-264-84:-84,], predictions8[::-1,w])[1][0] 
  accuracy_lasso[0,w] = 1

  #integrated_gradients=calculate_integrated_gradients_constbaseline(model, x_test, num_steps=P[i])
  #integrated_gradients=np.array(integrated_gradients.detach().numpy())

#print('integrated_gradients',integrated_gradients.shape)
for w in range(1):
    w=0
    plt.figure()
    plt.plot(skill[:,:],'y')
    plt.plot(np.mean(skill[:,:],axis=1),'r')
    plt.plot(accuracy_lasso[:,w],'b')#,label='Skill Lasso Reg')
    plt.plot(accuracy[:,w],'g')#,label='Skill Multilinear  Reg')
    plt.plot(persistence[w,::12],'k')
    plt.ylabel("skill (correlation values)",fontsize=13)
    plt.xlabel("Time (years)",fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    #plt.legend(fontsize='large', loc='lower left')
    plt.savefig('skill_total_NADW.png')


years=np.arange(1993,2015,1/12)

