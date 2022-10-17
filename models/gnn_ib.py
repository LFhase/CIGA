import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv, GCNConv, GINConv
from torch_geometric.utils import subgraph, to_dense_adj
import torch
from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import LeGNN
from utils.get_subgraph import relabel, split_batch, relabel_nodes
import numpy as np
from torch_geometric.utils import add_remaining_self_loops, degree

"""
Modified from 
Recognizing Predictive Substructures with Subgraph Information Bottleneck
https://arxiv.org/abs/2103.11155
"""

class GIB(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 mi_weight=0.3,
                 inner_lr=1e-3,
                 inner_wd=5e-5,
                 inner_loop=20,
                 JK="last",
                 graph_pooling="mean"):
        super(GIB, self).__init__()

        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio

        # used for attention
        self.att_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.Tanh(), nn.Linear(emb_dim * 2, 2))

        self.discriminator = Discriminator(input_size=emb_dim * 2, hidden_size=4 * emb_dim)
        self.inner_loop = inner_loop
        self.inner_lr = inner_lr
        self.inner_wd = inner_wd

        # used for classification
        self.cls_mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim * 2), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim * 2, emb_dim), torch.nn.ReLU())  # , torch.nn.ReLU()
        self.mse_criterion = torch.nn.CrossEntropyLoss(reduction='mean')  #torch.nn.MSELoss(reduction='mean')
        self.mseloss = torch.nn.MSELoss()

        self.EYE = torch.ones(2)
        self.mi_weight = mi_weight

        # self.reset_parameters()

    def cal_att(self, data, return_pos_penalty=False):
        # data.edge_index, _ = add_remaining_self_loops(data.edge_index, None, fill_value=1)
        h = self.gnn_encoder(data)

        self.EYE = self.EYE.to(h.device)

        atts = []
        group_features = []
        positives = []
        negatives = []
        graph_embeddings = []
        pos_penaltys = []

        batch_size = data.batch.max().item() + 1
        # remove 0-deg nodes that can appear in DrugOOD
        degs = degree(data.edge_index[0], num_nodes=h.size(0))
        # edge_indices, _, _, num_edges, cum_edges = split_batch(data)
        # for i,edges in enumerate(edge_indices):

        for i in range(batch_size):
            x_batch = h[(data.batch == i) & (degs != 0)]
            x_idx = torch.nonzero((data.batch == i) & (degs != 0), as_tuple=True)[0]

            edges, _ = subgraph(x_idx, data.edge_index, relabel_nodes=True)

            # 2 H
            att = F.softmax(self.att_mlp(x_batch), 1)  # self-att
            group_feature = torch.mm(torch.t(att), x_batch)
            graph_embedding = torch.mean(torch.mm(torch.t(att), x_batch), dim=0, keepdim=True)

            # calculate continuous loss
            Adj = to_dense_adj(edges)[0]
            Adj.requires_grad = False
            # print(Adj.size(), att.size())
            new_adj = torch.mm(torch.t(att), Adj)
            new_adj = torch.mm(new_adj, att)

            positive = torch.clamp(group_feature[0].unsqueeze(0), -100, 100)
            negative = torch.clamp(group_feature[1].unsqueeze(0), -100, 100)
            normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
            norm_diag = torch.diag(normalize_new_adj)
            pos_penalty = self.mseloss(norm_diag, self.EYE)

            atts.append(att)
            graph_embeddings.append(graph_embedding)
            group_features.append(group_feature)
            positives.append(positive)
            negatives.append(negative)
            pos_penaltys.append(pos_penalty)

        # graph_embedding = torch.mean(torch.mm(torch.t(att), x),dim=-2,keepdim=True)
        if return_pos_penalty:
            return graph_embeddings, positives, negatives, atts, h, pos_penaltys
        return graph_embeddings, positives, negatives, atts, h

    def forward(self, data, return_data="pred"):
        pred, postive = self.get_cls_results(data)
        if return_data.lower() == "rep":
            return pred, postive
        return pred

    def get_ib_loss(self, data, return_data="loss"):
        device = data.x.device
        labels = data.y

        graph_embeddings, positives, negatives, atts, x, pos_penaltys = self.cal_att(data, return_pos_penalty=True)
        embeddings = torch.cat(tuple(graph_embeddings), dim=0)
        positive = torch.cat(tuple(positives), dim=0)
        negative = torch.cat(tuple(negatives), dim=0)

        pos_penalty = torch.mean(torch.stack(tuple(pos_penaltys)))

        # calculate cls loss
        cls_input = torch.cat((embeddings, positive), dim=0)
        cls_labels = torch.cat((labels, labels), dim=0)

        cls_pred = self.cls_mlp(cls_input)
        cls_loss = self.mse_criterion(cls_pred, cls_labels)
        optimizer_local = torch.optim.Adam(self.discriminator.parameters(),
                                           lr=self.inner_lr,
                                           weight_decay=self.inner_wd)

        best_loss = 0
        cnt = 0
        # calculate MI loss
        for i in range(self.inner_loop):
            local_loss = -self.MI_Est(self.discriminator, embeddings, positive)
            optimizer_local.zero_grad()
            local_loss.backward(retain_graph=True)
            if local_loss < best_loss:
                best_loss = local_loss
                cnt = 0
            else:
                cnt += 1
            optimizer_local.step()
            if cnt >= 5:
                break
        mi_loss = self.MI_Est(self.discriminator, embeddings, positive)
        ib_loss = self.mi_weight * mi_loss + cls_loss + 5 * pos_penalty
        if return_data.lower() == "rep":
            return ib_loss, positive
        else:
            return ib_loss

    def get_cls_results(self, data):
        # device = data.x.device
        # labels = data.y
        graph_embeddings, positives, negatives, atts, x = self.cal_att(data)
        positive = torch.cat(tuple(positives), dim=0)
        cls_pred = self.cls_mlp(positive)
        return cls_pred, positive

    def MI_Est(self, discriminator, embeddings, positive):
        batch_size = embeddings.size(0)
        shuffle_embeddings = embeddings[torch.randperm(batch_size)]
        joint = discriminator(embeddings, positive)
        margin = discriminator(shuffle_embeddings, positive)
        valid_pos = torch.isfinite(torch.exp(margin))

        mi_est = torch.mean(joint[valid_pos]) - torch.log(torch.mean(torch.exp(margin[valid_pos])))
        return mi_est

    def get_causal_part(self, data, ib_loss=False):
        device = data.x.device
        graph_embeddings, positives, negatives, atts, x = self.forward(data)
        causal_edge_index = torch.LongTensor([[], []]).to(device)
        causal_edge_weight = torch.tensor([]).to(device)
        causal_edge_attr = torch.tensor([]).to(device)
        spu_edge_index = torch.LongTensor([[], []]).to(device)
        spu_edge_weight = torch.tensor([]).to(device)
        spu_edge_attr = torch.tensor([]).to(device)

        causal_nodes = torch.LongTensor([]).to(device)
        spu_nodes = torch.LongTensor([]).to(device)

        edge_indices, _, _, num_edges, cum_edges = split_batch(data)

        for (idx, (edge_index, N, C, att)) in enumerate(zip(edge_indices, num_edges, cum_edges, atts)):

            ind2orig = torch.nonzero(data.batch == idx, as_tuple=True)[0]
            _, ind = torch.max(att, 1)
            ind = ind.tolist()
            pos_ind = [i for i, j in enumerate(ind) if j == 0]
            neg_ind = [i for i, j in enumerate(ind) if j == 1]
            num_nodes = torch.unique(edge_index).size(0)
            flag = 0
            if len(pos_ind) == 0:
                pos_ind = np.random.randint(0, high=num_nodes, size=1).tolist()

            if len(neg_ind) == 0:
                flag = 1
                neg_ind = np.random.randint(0, high=num_nodes, size=1).tolist()

            c_edge, _ = subgraph(ind2orig[pos_ind], edge_index, relabel_nodes=False)
            s_edge, _ = subgraph(ind2orig[neg_ind], edge_index, relabel_nodes=False)

            causal_nodes = torch.cat([causal_nodes, ind2orig[pos_ind]])
            spu_nodes = torch.cat([spu_nodes, ind2orig[neg_ind]])
            causal_edge_index = torch.cat([causal_edge_index, c_edge], dim=1)
            spu_edge_index = torch.cat([spu_edge_index, s_edge], dim=1)

            causal_edge_weight = torch.cat([causal_edge_weight, torch.ones(c_edge.size(1), 1).to(device)])
            spu_edge_weight = torch.cat([spu_edge_weight, torch.ones(s_edge.size(1), 1).to(device)])

            causal_edge_attr = torch.cat([causal_edge_attr, data.edge_attr[ind2orig[pos_ind]]])
            spu_edge_attr = torch.cat([spu_edge_attr, data.edge_attr[ind2orig[neg_ind]]])

        causal_x, causal_edge_index, causal_batch, _ = relabel_nodes(x, causal_edge_index, causal_nodes, data.batch)
        spu_x, spu_edge_index, spu_batch, _ = relabel_nodes(x, spu_edge_index, spu_nodes, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch), None


class Discriminator(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        # self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, embeddings, positive):
        cat_embeddings = torch.cat((embeddings, positive), dim=-1)
        pre = self.relu(self.fc1(cat_embeddings))
        pre = self.fc2(pre)

        return pre

    def reset_parameters(self):
        torch.nn.init.constant_(self.fc1.weight, 0.01)
        torch.nn.init.constant_(self.fc2.weight, 0.01)
