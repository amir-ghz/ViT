import os
import json
import pandas as pd
import torch
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import importlib

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}_bs{}_lr{}_wd{}'.format(config.dataset, config.batch_size, config.lr, config.wd)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join('experiments', 'tb', exp_name)
    config.checkpoint_dir = os.path.join('experiments', 'save', exp_name, 'checkpoints/')
    config.result_dir = os.path.join('experiments', 'save', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.result_dir]:
        ensure_dir(dir)

    # save config
    write_json(vars(config), os.path.join('experiments', 'save', exp_name, 'config.json'))

    return config


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                print(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    if name == 'add_embedding':
                        add_data(tag=tag, mat=data, global_step=self.step, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr




#!/usr/bin/env python3
"""Converting between floats and binaries.
This code converts tensors of floats or bits into the respective other.
We use the IEEE-754 guideline [1] to convert. The default for conversion are
based on 32 bit / single precision floats: 8 exponent bits and 23 mantissa bits.
Other common formats are
num total bits     precision    exponent bits   mantissa bits       bias
---------------------------------------------------------------------------
    64 bits         double              11             52           1023
    32 bits         single               8             23            127
    16 bits         half                 5             10             15
Available modules:
    * bit2float
    * float2bit
    * integer2bit
    * remainder2bit
[1] IEEE Computer Society (2008-08-29). IEEE Standard for Floating-Point
Arithmetic. IEEE Std 754-2008. IEEE. pp. 1–70. doi:10.1109/IEEESTD.2008.4610935.
ISBN 978-0-7381-5753-5. IEEE Std 754-2008
Author, Karen Ullrich June 2019
"""

import torch
import warnings


def bit2float(b, num_e_bits=8, num_m_bits=23, bias=127.):
  """Turn input tensor into float.
      Args:
          b : binary tensor. The last dimension of this tensor should be the
          the one the binary is at.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 23.
          bias : Exponent bias/ zero offset. Default: 127.
      Returns:
          Tensor: Float tensor. Reduces last dimension.
  """
  b = b.to("cpu")
  expected_last_dim = num_m_bits + num_e_bits + 1
  assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                           "should be {}, not {}.".format(
    expected_last_dim, b.shape[-1])

  # check if we got the right type
  dtype = torch.float32
  if expected_last_dim > 32: dtype = torch.float64
  if expected_last_dim > 64:
    warnings.warn("pytorch can not process floats larger than 64 bits, keep"
                  " this in mind. Your result will be not exact.")
    
  s = torch.index_select(b, -1, torch.arange(0, 1))
  e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits))
  m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                             1 + num_e_bits + num_m_bits))
  # SIGN BIT
  out = ((-1) ** s).squeeze(-1).type(dtype)
  # EXPONENT BIT
  exponents = -torch.arange(-(num_e_bits - 1.), 1.)
  exponents = exponents.repeat(b.shape[:-1] + (1,))
  e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
  out *= 2 ** e_decimal
  # MANTISSA
  matissa = (torch.Tensor([2.]) ** (
    -torch.arange(1., num_m_bits + 1.))).repeat(
    m.shape[:-1] + (1,))
  out *= 1. + torch.sum(m * matissa, dim=-1)
  return out


def float2bit(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
  """Turn input tensor into binary.
      Args:
          f : float tensor.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 23.
          bias : Exponent bias/ zero offset. Default: 127.
          dtype : This is the actual type of the tensor that is going to be
          returned. Default: torch.float32.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  ## SIGN BIT
  s = torch.sign(f)
  f = f * s
  # turn sign into sign-bit
  s = (s * (-1) + 1.) * 0.5
  s = s.unsqueeze(-1)

  ## EXPONENT BIT
  e_scientific = torch.floor(torch.log2(f))
  e_decimal = e_scientific + bias
  e = integer2bit(e_decimal, num_bits=num_e_bits)

  ## MANTISSA
  m1 = integer2bit(f - f % 1, num_bits=num_e_bits)
  m2 = remainder2bit(f % 1, num_bits=bias)
  m = torch.cat([m1, m2], dim=-1)
  
  dtype = f.type()
  idx = torch.arange(num_m_bits).unsqueeze(0).type(dtype) \
        + (8. - e_scientific).unsqueeze(-1)
  idx = idx.long()
  m = torch.gather(m, dim=-1, index=idx)

  return torch.cat([s, e, m], dim=-1).type(dtype)


def remainder2bit(remainder, num_bits=127):
  """Turn a tensor with remainders (floats < 1) to mantissa bits.
      Args:
          remainder : torch.Tensor, tensor with remainders
          num_bits : Number of bits to specify the precision. Default: 127.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = remainder.type()
  exponent_bits = torch.arange(num_bits).type(dtype)
  exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
  out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
  return torch.floor(2 * out)


def integer2bit(integer, num_bits=8):
  """Turn integer tensor to binary representation.
      Args:
          integer : torch.Tensor, tensor with integers
          num_bits : Number of bits to specify the precision. Default: 8.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = integer.type()
  exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
  exponent_bits = exponent_bits.repeat(integer.shape + (1,))
  out = integer.unsqueeze(-1) / 2 ** exponent_bits
  return (out - (out % 1)) % 2