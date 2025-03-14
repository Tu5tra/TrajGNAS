import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.nn.parameter import Parameter
################################
###########   ST-GCN   #########
################################
class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class STGCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(120, out_features).cuda())
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters ( self ):
        stdv = 1. / math.sqrt ( self.weight.size ( 1 ) )
        self.weight.data.uniform_ ( -stdv , stdv )
        if self.bias is not None:
            self.bias.data.uniform_ ( -stdv , stdv )

    def forward ( self , input , adj ):
        support = torch.mm ( input , self.weight )
        output = torch.spmm ( adj , support )
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid,dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, A):
        x = F.relu(self.gc1(x, A))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, A)
        # return F.log_softmax(x, dim=1)

        return x, A


################################
###########   GAT   ############
################################

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        #初始化权重
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input, adj):
        adj=adj.unsqueeze(0).expand(input.shape[0], -1, -1, -1)
        h_prime_cat = torch.zeros(size=(input.shape[0],
                                        input.shape[2],
                                        input.shape[3],
                                        self.out_features)).to(input.device)

        for step_i in range(input.shape[2]):
            input_i = input[:, :, step_i, :]
            input_i = input_i.permute(0, 2, 1)


            adj_i = adj.mean(dim=1)
            Wh = torch.matmul(input_i, self.W)

            batch_size = Wh.size()[0]
            N = Wh.size()[1]  # number of nodes
            Wh_chunks = Wh.repeat(1, 1, N).view(batch_size, N * N, self.out_features)
            Wh_alternating = Wh.repeat(1, N, 1)
            combination_matrix = torch.cat([Wh_chunks, Wh_alternating], dim=2)
            a_input = combination_matrix.view(batch_size, N, N, 2 * self.out_features)

            #计算注意力系数
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)

            attention = torch.where(adj_i > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, 0.25, training=self.training)
            #计算节点特征
            h_prime = torch.matmul(attention, Wh)  # [8, 120, 64]
            h_prime_cat[:, step_i, :, :] = h_prime

        if self.concat:
            return F.elu(h_prime_cat)
            # return h_prime_return
        else:
            return h_prime_cat


class GATBlock(nn.Module):
    def __init__(self, input_dim, out_channels, stride=1, residual=True):
        super(GATBlock, self).__init__()

        self.att_1 = GraphAttentionLayer(input_dim, out_channels, concat=True)
        self.att_2 = GraphAttentionLayer(input_dim, out_channels, concat=True)
        self.att_out = GraphAttentionLayer(out_channels, out_channels, concat=False)

        if not residual:
            self.residual = lambda x: 0
        elif (input_dim == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels), )

    def forward(self, x, adjacency):
        res = self.residual(x)
        x_1 = self.att_1(x, adjacency)
        x_2 = self.att_2(x, adjacency)
        x = torch.stack([x_1, x_2], dim=-1).mean(-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.dropout(x, 0.25)
        x = F.elu(self.att_out(x, adjacency))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x + res
        return x , adjacency

class GAT(nn.Module):
    def __init__(self, input_dim, out_channels):
        super(GAT, self).__init__()

        self.att = GraphAttentionLayer(input_dim, out_channels, concat=True)


    def forward(self, x, adjacency):
        x = F.elu(self.att(x ,adjacency))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.dropout(x, 0.25)
        return x,adjacency
