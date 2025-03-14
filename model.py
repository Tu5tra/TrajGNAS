import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
import numpy as np
from graph import Graph


from metrics import *
from utils import * 
from utils_model import *

from LOSS import region_loss, compute_cost_loss , count_parameters

def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")

class NaOp(nn.Module):
  def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
    super(NaOp, self).__init__()

    self._op = NA_OPS[primitive](in_dim, out_dim)
    self.op_linear = nn.Linear(in_dim, out_dim)
    self.act = act_map(act)
    self.with_linear = with_linear

  def forward(self, x, edge_index):
      output1 = self._op(x, edge_index)

      # 检查 output 的类型
      if isinstance(output1, tuple):
          output1 = output1[0]  

      if self.with_linear:
          return self.act(output1 + self.op_linear(x))
      else:
          return self.act(output1)
# class NaMLPOp(nn.Module):
#     def __init__(self, primitive, in_dim, out_dim, act):
#         super(NaMLPOp, self).__init__()
#         self._op = NA_MLP_OPS[primitive](in_dim, out_dim)
#         self.act = act_map(act)
# 
#     def forward(self, x, edge_index):
#         return self.act(self._op(x, edge_index))

class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)

class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act, num_layers=None):
        super(LaOp, self).__init__()
        self._op = LA_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))
    
class node_o(nn.Module):
    def __init__(self, device='cuda:0'):
        super(node_o, self).__init__()
        self.device = device  # 添加设备参数，默认为 'cuda:0'

    def forward(self, a):
        # 确保输入数据在 GPU 上
        a = a.to(self.device)

        node = []
        node_64 = []
        for q in range(1, a.shape[0]):
            node_single = []
            node_single_64 = []
            for qq in range(a[q].shape[0]):
                for qqq in range(a[q].shape[0]):
                   
                    dis=torch.sqrt((a[q][qq][0]-a[q][qqq][0])*(a[q][qq][0]-a[q][qqq][0])+(a[q][qq][1]-a[q][qqq][1])*(a[q][qq][1]-a[q][qqq][1]))
                    d1=torch.sqrt((a[q][qq][0]-a[q-1][qq][0])*(a[q][qq][0]-a[q-1][qq][0])+(a[q][qq][1]-a[q-1][qq][1])*(a[q][qq][1]-a[q-1][qq][1]))
                    v1=d1/0.5
                    d2=torch.sqrt((a[q][qqq][0]-a[q-1][qqq][0])*(a[q][qqq][0]-a[q-1][qqq][0])+(a[q][qqq][1]-a[q-1][qqq][1])*(a[q][qqq][1]-a[q-1][qqq][1]))
                    v2=d2/0.5

                    angle1 = angle_l([a[q-1, qq], a[q, qq]])  
                    angle2 = angle_l([a[q-1, qqq], a[q, qqq]])

                    x_linju=[a[q][qq][0]]
                    y_linju=[a[q][qq][1]]
                    v_linju=[v1]
                    angle_linju = [angle1]
                    if dis <= 12:
                        x_linju.append(a[q][qqq][0])
                        y_linju.append(a[q][qqq][1])
                        v_linju.append(v2.item())
                        angle_linju.append(angle2)

                    # 初始化 MLP 并在 GPU 上运行
                    mlp1 = MLP(len(x_linju), 1).to(self.device)
                    mlp2 = MLP(len(x_linju), 1).to(self.device)
                    mlp3 = MLP(len(x_linju), 1).to(self.device)
                    mlp4 = MLP(len(x_linju), 1).to(self.device)
                    mlp5 = MLP(4, 1).to(self.device)
                    mlp6 = MLP(4, 64).to(self.device)

                    # MLP 输入和输出处理 
                    x_mlp = mlp1(torch.tensor(x_linju, device=self.device, dtype=torch.float))
                    y_mlp = mlp2(torch.tensor(y_linju, device=self.device, dtype=torch.float))
                    v_mlp = mlp3(torch.tensor(v_linju, device=self.device, dtype=torch.float))
                    angle_mlp = mlp4(torch.tensor(angle_linju, device=self.device, dtype=torch.float))

                    sss = mlp5(torch.tensor([x_mlp[0], y_mlp[0], v_mlp[0], angle_mlp[0]], device=self.device))
                    sss_64 = mlp6(torch.tensor([x_mlp[0], y_mlp[0], v_mlp[0], angle_mlp[0]], device=self.device))

                node_single.append(sss.tolist())
                node_single_64.append(sss_64.tolist())
            node.append(node_single)
            node_64.append(node_single_64)

        
        node_out = [node[0]]
        for jj in node:
            node_out.append(jj)
        node_out1 = torch.tensor(node_out, dtype=torch.float32, device=self.device)

        
        node_out_64 = [node_64[0]]
        for jj in node_64:
            node_out_64.append(jj)
        node_out2_64 = torch.tensor(node_out_64, dtype=torch.float32, device=self.device)

        return node_out1, node_out2_64


class risk_interaction(nn.Module):
    def __init__(self,device='cuda:0'):
        super(risk_interaction,self).__init__() 
        self.device = device
        self.node_o = node_o()
        self.mlp = MLP(2,1)
        self.to(self.device) 
        
    def forward(self,a,start,end,sa_out,se_out,pedestrian_index,obs_traj_type):
        a=a.permute(2,0,1)
        # clu=cluster[start:end]

        scene_graph_a=torch.cat((a, sa_out[:,:,-2:]), 1)   
        scene_graph_e=torch.zeros((scene_graph_a.shape[0],scene_graph_a.shape[1],scene_graph_a.shape[1]))
        scene_graph_a.to(self.device)
        

        for p in range(se_out.shape[0]):
            for pp in range(se_out.shape[1]):
                for ppp in range(se_out.shape[1]):
                    scene_graph_e[p][pp][ppp]=se_out[p][pp][ppp]

        ped_ii=[]   
        veh_ii=[]
        for ii,ind in enumerate(range(start,end)):
            if ind in pedestrian_index:
                ped_ii.append(ii)
            else:
                veh_ii.append(ii)


        node_ou,node_ou_64=self.node_o(a)
        risk_inter=[]
        for k in range(1,a.shape[0]):
            for kk in range(sa_out.shape[1]):
               for e in ped_ii:
                   if sa_out[k][kk][4]==1.0 or sa_out[k][kk][5]==1.0:
                       scene_graph_e[k][kk][kk+e]=scene_graph_e[k][kk][kk+e]=1
               for ee in ped_ii:
                   if sa_out[k][kk][0]==1.0 or sa_out[k][kk][1]==1.0:
                       scene_graph_e[k][kk][kk+ee]=scene_graph_e[k][kk][kk+ee]=1
               risk_inter1=torch.zeros((a.shape[1],a.shape[1]), device=self.device)
               risk_inter1[a.shape[1]-1,a.shape[1]-1]=0
               

               for i in range(a.shape[1]):
                    if i in ped_ii :#对于行人
                       
        
                       if (sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) < a[k][i][0] < (
                               sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) or (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2) < a[k][i][1] < (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2):
                           

                           if sa_out[k][kk][0] == 1.0 or sa_out[k][kk][1] == 1.0 or sa_out[k][kk][4] == 1.0: 
                               for j in range(a.shape[1]):
                                   if i == j:
                                       risk_inter1[i, i] = 0
                                   else:
                                       d1 = torch.sqrt((a[k][i][0] - a[k - 1][i][0]) * (a[k][i][0] - a[k - 1][i][0]) + (
                                                   a[k][i][1] - a[k - 1][i][1]) * (a[k][i][1] - a[k - 1][i][1]))
                                       v1 = d1 / 0.5
                                       d2 = torch.sqrt((a[k][j][0] - a[k - 1][j][0]) * (a[k][j][0] - a[k - 1][j][0]) + (
                                                   a[k][j][1] - a[k - 1][j][1]) * (a[k][j][1] - a[k - 1][j][1]))
                                       v2 = d2 / 0.5
                                       angle1 = angle_l([a[k - 1][i], a[k][i]])
                                       angle2 = angle_l([a[k - 1][j], a[k][j]])
                                       dis = torch.sqrt((a[k][j][0] - a[k][i][0]) * (a[k][j][0] - a[k][i][0]) + (
                                                   a[k][j][1] - a[k][i][1]) * (a[k][j][1] - a[k][i][1]))
                                       angle3 = angle_l([a[k][i], a[k][j]])
                                       if (angle1 - np.pi / 2) < angle3 < (angle1 + np.pi / 2):
                                            if obs_traj_type[i]==1 or obs_traj_type[i]==2:
                                                lij = 0.45
                                            if obs_traj_type[i]==3:
                                                lij = 0.9
                                            if obs_traj_type[i]==4:
                                                lij = 0.65
                                            else:
                                                lij = 1
                                       else:
                                           lij = 0
                                       vv = abs(v1 * math.cos(abs(angle1 - angle3)) - v2 * math.cos(abs(angle2 - angle3)))
                                       t = dis / vv
                                       risk1 = 1 / t
#                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
                                      
                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
#                                       risk=bb*risk1
                                       risk = risk1 * bb * lij
                                       risk_inter1[i, j] = risk
                       else:
                            for j in range(len(ped_ii)):

                                if i==j:
                                    risk_inter1[i,i]=0
                                else:
                                    d1=torch.sqrt((a[k][i][0]-a[k-1][i][0])*(a[k][i][0]-a[k-1][i][0])+(a[k][i][1]-a[k-1][i][1])*(a[k][i][1]-a[k-1][i][1]))
                                    v1=d1/0.5
                                    d2=torch.sqrt((a[k][j][0]-a[k-1][j][0])*(a[k][j][0]-a[k-1][j][0])+(a[k][j][1]-a[k-1][j][1])*(a[k][j][1]-a[k-1][j][1]))
                                    v2=d2/0.5
                                    
                                    angle1=angle_l([a[k-1][i],a[k][i]])
                                    angle2=angle_l([a[k-1][j],a[k][j]])
                        
                                    dis=torch.sqrt((a[k][j][0]-a[k][i][0])*(a[k][j][0]-a[k][i][0])+(a[k][j][1]-a[k][i][1])*(a[k][j][1]-a[k][i][1]))
                                    angle3=angle_l([a[k][i],a[k][j]])
                                    if (angle1- math.pi/2)<angle3<(angle1+math.pi/2): 
                                        if obs_traj_type[i]==1 or obs_traj_type[i]==2:
                                            lij = 0.45
                                        if obs_traj_type[i]==3:
                                            lij = 0.9
                                        if obs_traj_type[i]==4:
                                            lij = 0.65
                                        else:
                                            lij = 1
                                    else:
                                        lij=0
                                    vv=abs(v1*math.cos(abs(angle1-angle3))-v2*math.cos(abs(angle2-angle3)))
                                    t=dis/vv
                                    risk1=1/t

                                    bb = self.mlp(torch.tensor([node_ou[k][i] , node_ou[k][j] ], device=self.device, dtype=torch.float))
                                    risk=risk1*bb*lij                        
                                    risk_inter1[i,j]=risk
                    
                    elif i in veh_ii:
                        if (sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) < a[k][i][0] < (
                               sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) or (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2) < a[k][i][1] < (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2):
                           

                           if sa_out[k][kk][0] == 1.0 or sa_out[k][kk][1] == 1.0 or sa_out[k][kk][4] == 1.0: #检查是否有边连接
                               for j in range(a.shape[1]):
                                   if i == j:
                                       risk_inter1[i, i] = 0
                                   else:
                                       d1 = torch.sqrt((a[k][i][0] - a[k - 1][i][0]) * (a[k][i][0] - a[k - 1][i][0]) + (
                                                   a[k][i][1] - a[k - 1][i][1]) * (a[k][i][1] - a[k - 1][i][1]))
                                       v1 = d1 / 0.5
                                       d2 = torch.sqrt((a[k][j][0] - a[k - 1][j][0]) * (a[k][j][0] - a[k - 1][j][0]) + (
                                                   a[k][j][1] - a[k - 1][j][1]) * (a[k][j][1] - a[k - 1][j][1]))
                                       v2 = d2 / 0.5
                                       angle1 = angle_l([a[k - 1][i], a[k][i]])
                                       angle2 = angle_l([a[k - 1][j], a[k][j]])
                                       dis = torch.sqrt((a[k][j][0] - a[k][i][0]) * (a[k][j][0] - a[k][i][0]) + (
                                                   a[k][j][1] - a[k][i][1]) * (a[k][j][1] - a[k][i][1]))
                                       angle3 = angle_l([a[k][i], a[k][j]])
                                       if (angle1 - np.pi / 2) < angle3 < (angle1 + np.pi / 2):
                                            if obs_traj_type[i]==1 or obs_traj_type[i]==2:
                                                lij = 0.45
                                            if obs_traj_type[i]==3:
                                                lij = 0.9
                                            if obs_traj_type[i]==4:
                                                lij = 0.65
                                            else:
                                                lij = 1
                                       else:
                                           lij = 0
                                       vv = abs(v1 * math.cos(abs(angle1 - angle3)) - v2 * math.cos(abs(angle2 - angle3)))
                                       t = dis / vv
                                       risk1 = 1 / t
#                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
                                      
                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
#                                       risk=bb*risk1
                                       risk = risk1 * bb * lij
                                       risk_inter1[i, j] = risk
                        else:
                            for j in range(len(ped_ii)):

                                if i==j:
                                    risk_inter1[i,i]=0
                                else:
                                    d1=torch.sqrt((a[k][i][0]-a[k-1][i][0])*(a[k][i][0]-a[k-1][i][0])+(a[k][i][1]-a[k-1][i][1])*(a[k][i][1]-a[k-1][i][1]))
                                    v1=d1/0.5
                                    d2=torch.sqrt((a[k][j][0]-a[k-1][j][0])*(a[k][j][0]-a[k-1][j][0])+(a[k][j][1]-a[k-1][j][1])*(a[k][j][1]-a[k-1][j][1]))
                                    v2=d2/0.5
                                    
                                    angle1=angle_l([a[k-1][i],a[k][i]])
                                    angle2=angle_l([a[k-1][j],a[k][j]])
                        
                                    dis=torch.sqrt((a[k][j][0]-a[k][i][0])*(a[k][j][0]-a[k][i][0])+(a[k][j][1]-a[k][i][1])*(a[k][j][1]-a[k][i][1]))
                                    angle3=angle_l([a[k][i],a[k][j]])
                                    if (angle1- math.pi/2)<angle3<(angle1+math.pi/2): 
                                        if obs_traj_type[i]==1 or obs_traj_type[i]==2:
                                            lij = 0.45
                                        if obs_traj_type[i]==3:
                                            lij = 0.9
                                        if obs_traj_type[i]==4:
                                            lij = 0.65
                                        else:
                                            lij = 1
                                    else:
                                        lij=0
                                    vv=abs(v1*math.cos(abs(angle1-angle3))-v2*math.cos(abs(angle2-angle3)))
                                    t=dis/vv
                                    risk1=1/t

                                    bb = self.mlp(torch.tensor([node_ou[k][i] , node_ou[k][j] ], device=self.device, dtype=torch.float))
                                    risk=risk1*bb*lij                        
                                    risk_inter1[i,j]=risk



            risk_inter.append(risk_inter1)

        risk_inter_out1 = [risk_inter[0]] + risk_inter
        risk_inter_out = torch.stack(risk_inter_out1, dim=0)
        scene_graph_e = scene_graph_e.to(self.device)
        return risk_inter_out, scene_graph_a,scene_graph_e,node_ou_64    
class NetworkGNN(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self, genotype, criterion, in_dim, hidden_size,graph_args,batch_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(NetworkGNN, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers  #图网络层数
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self._criterion = criterion   #损失函数
        ops = genotype.split('||')
        self.args = args
        self.risk_interaction=risk_interaction()

        self.graph = Graph(**graph_args)
        self.A = np.ones((graph_args['max_hop'] + 1, graph_args['num_node'], graph_args['num_node']))
        self.num_node = num_node = self.graph.num_node
        self.out_dim_per_node = out_dim_per_node = 2  # (x, y) coordinate

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(4, 12, 3, padding=1))
        for j in range(1, 5):
            self.tpcnns.append(nn.Conv2d(12, 12, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(12, 12, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(5):
            self.prelus.append(nn.PReLU())
        
       

       
        self.lin1 = nn.Linear(in_dim, hidden_size)  

        
        self.gnn_layers = nn.ModuleList(
                [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear) for i in range(num_layers)])



        
        if self.args.fix_last:
            if self.num_layers > 1:
                self.sc_layers = nn.ModuleList([ScOp(ops[i+num_layers]) for i in range(num_layers - 1)])
            else:
                self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
        else:
            # no output conditions.
            skip_op = ops[num_layers:2 * num_layers]
            if skip_op == ['none'] * num_layers:
                skip_op[-1] = 'skip'
                print('skip_op:', skip_op)
            self.sc_layers = nn.ModuleList([ScOp(skip_op[i]) for i in range(num_layers)])

    
        self.lns = torch.nn.ModuleList()
        if self.args.with_layernorm:
            for i in range(num_layers):
                self.lns.append(LayerNorm(hidden_size, affine=True))

        self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers) 



    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new


 
    def forward(self, obs_traj,obs_traj_rel,pred_traj_gt,start,pred_traj_gt_rel,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index,obs_traj_type):
      
      risk_out,sg_a,sg_e,node_ou_64=self.risk_interaction(obs_traj[start:end,:],start,end,sa_out,se_out,pedestrian_index,obs_traj_type)
      node_ou_64=node_ou_64.unsqueeze(0)
      node_ou_64=node_ou_64.permute(0,3,1,2)
      norm_lap_matr=True
      v_obs,a_ = seq_to_graph(obs_traj[start:end,:],obs_traj_rel[start:end, :],norm_lap_matr)
      V_obs=v_obs.unsqueeze(0)
      V_obs_tmp =V_obs.permute(0,3,1,2)

     
      js = []
      for i in range(self.num_layers):
          x = self.gnn_layers[i](node_ou_64, risk_out)
          if self.args.with_layernorm:
                # layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
                # x = layer_norm(x)
                x = self.lns[i](x)
          x = F.dropout(x, p=self.in_dropout, training=self.training)
          if i == self.num_layers - 1 and self.args.fix_last:
                js.append(x)
          else:
                js.append(self.sc_layers[i](x))
      x5 = self.layer6(js)
      x5 = F.dropout(x5, p=self.out_dropout, training=self.training)
      
      # Merge features
      merge_feature = x5
      v1=merge_feature
      v = v1.view(v1.shape[0], v1.shape[2], v1.shape[1], v1.shape[3])
      v = self.prelus[0](self.tpcnns[0](v))
      for k in range(1, 4):
          v = self.prelus[k](self.tpcnns[k](v)) + v
      v = self.tpcnn_ouput(v)
      v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
      return v

    def _loss(self, batch_val):
        loss_batch = 0 
        batch_count = 0
        is_fst_loss = True

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,loss_mask,seq_start_end,sa,se,pedestrian_index,vehicle_index,rider_index,obs_traj_type= batch_val
        self.cuda()
        obs_traj = obs_traj.cuda()
        pred_traj_gt = pred_traj_gt.cuda()
        obs_traj_rel = obs_traj_rel.cuda()
        pred_traj_gt_rel = pred_traj_gt_rel.cuda()
        seq_start_end = torch.tensor(seq_start_end).cuda()
        pedestrian_index = torch.tensor(pedestrian_index).cuda()
        vehicle_index = torch.tensor(vehicle_index).cuda()
        rider_index = torch.tensor(rider_index).cuda()
        obs_traj_type = obs_traj_type.cuda()
        sa = sa.cuda()
        se = se.cuda()
        obs_len=4

        loader_len=len(seq_start_end)
        
    
        obs_traj=np.squeeze(obs_traj,axis=0)     
        pred_traj_gt=np.squeeze(pred_traj_gt,axis=0)
        obs_traj_rel=np.squeeze(obs_traj_rel,axis=0)
        pred_traj_gt_rel=np.squeeze(pred_traj_gt_rel,axis=0)
        seq_start_end=torch.tensor(seq_start_end)
        turn_point =int(loader_len/self.batch_size)*self.batch_size+ loader_len%self.batch_size -1
        index=[i for i in range(len(seq_start_end))]
        random.shuffle(index)
        seq_start_end=seq_start_end[index]

        batch_count = 0
        for ss in range(len(seq_start_end)):
          batch_count+=1
          cnt=ss
          start, end = seq_start_end[ss]
          norm_lap_matr=True
          v_tr,a_=seq_to_graph(pred_traj_gt[start:end,:],pred_traj_gt_rel[start:end, :],norm_lap_matr)
          se_out=se[ss,0:obs_len,:]
          sa_out=sa[ss,0:obs_len,:]

          V_pred = self(obs_traj,obs_traj_rel,pred_traj_gt,start,pred_traj_gt_rel,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index,obs_traj_type)
          V_pred = V_pred.permute(0,2,3,1)
          V_pred = V_pred.squeeze()
          V_tr = v_tr
          
       
          graph_dis_loss = graph_loss(V_pred, V_tr)
        
          
          region_check_loss = region_loss(V_pred,obs_len, obs_traj, start,end,sa_out)

         
          eff_loss = compute_cost_loss(self, flops_weight=1e-6, sparsity_weight=1e-5)


          graph_loss_weight = 1
          efficiency_loss_weight = 0.7
          region_loss_weight = 0.3 

          total_loss = graph_loss_weight*graph_dis_loss + region_loss_weight * region_check_loss + efficiency_loss_weight * eff_loss

          if batch_count%self.batch_size !=0 and cnt != turn_point :
              
              if is_fst_loss :
                  loss = total_loss
                  is_fst_loss = False
              else:
                  loss += total_loss

          else:
              loss = loss/self.batch_size
          loss_batch += loss.item()
          loss_all = loss_batch/batch_count
          loss_all1 = torch.tensor(loss_all, dtype=torch.float32, requires_grad=True)

        return loss_all1


    def arch_parameters(self):
        return self._arch_parameters


    #使用 softmax 将架构参数转换为权重，并通过权重来解析出最终的基因型 genotype。
    def genotype(self):

        def _parse(na_weights, sc_weights, la_weights):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(NA_PRIMITIVES[k])
            #sc_indices = sc_weights.argmax(dim=-1)
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(SC_PRIMITIVES[k])
            #la_indices = la_weights.argmax(dim=-1)
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(LA_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(), F.softmax(self.la_alphas, dim=-1).data.cpu())

        return gene

    def reshape_for_lstm(self, feature):
        # prepare for skeleton prediction model
        '''
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        '''
        N, C, T, V = feature.size()
        now_feat = feature.permute(0, 3, 2, 1).contiguous()  # to (N, V, T, C)
        now_feat = now_feat.view(N * V, T, C)
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(-1, self.num_node, T,
                                  self.out_dim_per_node)  # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous()  # (N, C, T, V)
        return now_feat


