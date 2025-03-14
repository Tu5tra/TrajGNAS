import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU

from tp_gcn_layers import ConvTemporalGraphical , STGCN ,GCN ,GAT, GATBlock

#TODO 完善所有的节点聚合器目录
NA_OPS = {
    'gcn': lambda in_channels, out_channels: NaAggregator(in_channels, out_channels, 'gcn'),
    't-gcn': lambda in_channels, out_channels: NaAggregator(in_channels, out_channels, 't-gcn'),
    'st-gcn': lambda in_channels, out_channels: NaAggregator(in_channels, out_channels, 'st-gcn'),
    'gat': lambda in_channels, out_channels: NaAggregator(in_channels, out_channels, 'gat'),
    'ai-gat': lambda in_channels, out_channels: NaAggregator(in_channels, out_channels, 'ai-gat'),
}

SC_OPS={
  'none': lambda: Zero(),
  'skip': lambda: Identity(),
  }

LA_OPS={
  'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
  'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
  'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers),
  'l_sum': lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
  'l_mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers)
}

class NaAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, aggregator):
        super(NaAggregator, self).__init__()

        #TODO 完善所有的节点聚合器操作

        if 'gcn' == aggregator:
            nhid = 16
            self._op = GCN(in_channels, out_channels, nhid ,dropout =0.5)
        if 't-gcn' == aggregator:
            self._op = ConvTemporalGraphical(in_channels, out_channels, kernel_size = 4, bias=True)
        if 'st-gcn' == aggregator:
            spatial_kernel_size = 4  
            temporal_kernel_size = 3  
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self._op = STGCN(in_channels, out_channels, kernel_size)
        if 'gat' == aggregator:
            self._op = GAT(in_channels, out_channels)
        if 'ai-gat' == aggregator:
            self._op = GATBlock(in_channels, out_channels, stride=1, residual=True)

    def forward(self, x, A):
        return self._op(x, A)


class LaAggregator(nn.Module):

  def __init__(self, mode, hidden_size, num_layers=3):
    super(LaAggregator, self).__init__()
    self.mode = mode
    if mode in ['lstm', 'cat', 'max']:
      self.jump = JumpingKnowledge(mode, channels=hidden_size, num_layers=num_layers)
    elif mode == 'att':
      self.att = Linear(hidden_size, 1)

    if mode == 'cat':
        self.lin = Linear(hidden_size * num_layers, hidden_size)
    else:
        self.lin = Linear(hidden_size, hidden_size)

  def forward(self, xs):
    if self.mode in ['lstm', 'cat', 'max']:
      output = self.jump(xs)
    elif self.mode == 'sum':
      output = torch.stack(xs, dim=-1).sum(dim=-1)
    elif self.mode == 'mean':
      output = torch.stack(xs, dim=-1).mean(dim=-1)
    elif self.mode == 'att':
      input = torch.stack(xs, dim=-1).transpose(1, 2)
      weight = self.att(input)
      weight = F.softmax(weight, dim=1)
      output = torch.mul(input, weight).transpose(1, 2).sum(dim=-1)

    # return self.lin(F.relu(self.jump(xs)))
    return F.relu(output)



class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):

  def __init__(self):
    super(Zero, self).__init__()

  def forward(self, x):
    return x.mul(0.)