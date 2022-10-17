import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GlobalAttention, LEConv, Set2Set,
                                global_add_pool, global_max_pool,
                                global_mean_pool)

from models.conv import GNN_node, GNN_node_Virtualnode


class GNN(torch.nn.Module):

    def __init__(self,
                 num_class,
                 num_layer=5,
                 emb_dim=300,
                 input_dim=1,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 pred_head="cls",
                 edge_dim=-1):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_node = LeGNN(in_channels=input_dim,
                                  hid_channels=emb_dim,
                                  num_layer=num_layer,
                                  drop_ratio=drop_ratio,
                                  num_classes=num_class,
                                  edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_node = GNN_node_Virtualnode(num_layer,
                                                     emb_dim,
                                                     input_dim=input_dim,
                                                     JK=JK,
                                                     drop_ratio=drop_ratio,
                                                     residual=residual,
                                                     gnn_type=gnn_type,
                                                     edge_dim=edge_dim)
            else:
                self.gnn_node = GNN_node(num_layer,
                                         emb_dim,
                                         input_dim=input_dim,
                                         JK=JK,
                                         drop_ratio=drop_ratio,
                                         residual=residual,
                                         gnn_type=gnn_type,
                                         edge_dim=edge_dim)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(torch.nn.Linear(
                emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 *
                                                            emb_dim), torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if pred_head == "cls":
            if graph_pooling == "set2set":
                self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_class)
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)
        elif pred_head == "inv":
            self.graph_pred_linear = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                                         nn.Linear(2 * emb_dim, self.num_class))

            self.spu_mlp = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                                nn.Linear(2 * emb_dim, self.num_class))
            self.cq = nn.Linear(self.num_class, self.num_class)
            self.spu_fw = torch.nn.Sequential(self.spu_mlp, self.cq)
        elif pred_head == "spu":
            self.graph_pred_linear = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                                         nn.Linear(2 * emb_dim, self.num_class))
            self.spu_gcn = GNN_node(num_layer=1,
                                     emb_dim=emb_dim,
                                     input_dim=emb_dim,
                                     JK=JK,
                                     drop_ratio=drop_ratio,
                                     residual=residual,
                                     gnn_type=gnn_type)
            self.spu_mlp = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                                nn.Linear(2 * emb_dim, self.num_class))
            self.cq = nn.Linear(self.num_class, self.num_class)
            self.spu_fw = torch.nn.Sequential(self.spu_mlp, self.cq)

    def get_spu_pred_forward(self, batched_data, get_rep=False):
        # if using DIR, won't consider gradients for encoder
        # h_node = self.gnn_node(batched_data)
        # h_graph = self.pool(h_node, batched_data.batch).detach()
        h_node = self.spu_gcn(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        if get_rep:
            return self.spu_fw(h_graph), h_graph
        return self.spu_fw(h_graph)

    def get_spu_pred(self, batched_data, get_rep=False):
        # if using DIR, won't consider gradients for encoder
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch).detach()

        if get_rep:
            return self.spu_fw(h_graph), h_graph
        return self.spu_fw(h_graph)

    def forward(self, batched_data, get_rep=False):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        if get_rep:
            return self.graph_pred_linear(h_graph), h_graph
        return self.graph_pred_linear(h_graph)

    def forward_rep(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph

    def forward_cls(self, h_graph):
        return self.graph_pred_linear(h_graph)

    def forward_spu_cls(self, h_graph):
        return self.spu_fw(h_graph)

    def forward_cl(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        z = self.proj_head(h_graph)
        return z

    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()
        return loss


class LeGNN(torch.nn.Module):

    def __init__(self, in_channels, hid_channels=64, num_classes=3, num_layer=2, drop_ratio=0.5, edge_dim=-1):
        super().__init__()

        self.num_layer = num_layer
        self.node_emb = nn.Linear(in_channels, hid_channels)
        self.drop_ratio = drop_ratio
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(num_layer):
            conv = LEConv(in_channels=hid_channels, out_channels=hid_channels)
            self.convs.append(conv)
            self.relus.append(nn.ReLU())

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        return node_x

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        for conv, ReLU in zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)
            x = ReLU(x)
        node_x = x
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):

        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_causal_pred(self, causal_graph_x):
        pred = self.causal_mlp(causal_graph_x)
        return pred

    def get_spu_pred(self, spu_graph_x):
        pred = self.spu_fw(spu_graph_x)
        return pred

    def get_comb_pred(self, causal_graph_x, spu_graph_x):
        causal_pred = self.causal_mlp(causal_graph_x)
        spu_pred = self.spu_mlp(spu_graph_x).detach()
        return torch.sigmoid(spu_pred) * causal_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

if __name__ == '__main__':
    GNN(num_class=10)
