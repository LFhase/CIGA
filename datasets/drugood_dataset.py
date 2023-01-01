import os.path as osp
import pickle as pkl

import torch
import random
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, add_self_loops


class DrugOOD(InMemoryDataset):

    def __init__(self, root, dataset, name, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DrugOOD dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data(root, dataset, name, mode)

    def load_data(self, root, dataset, name, mode):
        data_path = osp.join(root, name + "_" + mode + ".pt")
        if not osp.exists(data_path):
            data_list = []
            # for data in dataset:
            for step, data in tqdm(enumerate(dataset), total=len(dataset), desc="Converting"):
                graph = data['input']
                y = data['gt_label']
                group = data['group']

                edge_index = graph.edges()
                edge_attr = graph.edata['x']  #.long()
                node_attr = graph.ndata['x']  #.long()
                new_data = Data(edge_index=torch.stack(list(edge_index), dim=0),
                                edge_attr=edge_attr,
                                x=node_attr,
                                y=y,
                                group=group)
                data_list.append(new_data)
            torch.save(self.collate(data_list), data_path)

        self.data, self.slices = torch.load(data_path)
