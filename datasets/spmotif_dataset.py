import os.path as osp
import pickle as pkl

import torch
import random
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, add_self_loops


class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        print(f"######################################## {root}")
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'raw', 'SPMotif_train.npy')):
            print("raw data of `SPMotif` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(osp.join(
            self.raw_dir, self.raw_file_names[idx]),
                                                                                    allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, ground_truth, z,
                  p) in enumerate(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            x[index, z] = 1
            if 'mspmotif' in self.root.lower():
                # additionally add the spuriously correlated node features
                bias = 0.5
                if '0.3' in self.root:
                    bias = 0.33
                elif '0.4' in self.root:
                    bias = 0.4
                elif '0.5' in self.root:
                    bias = 0.5
                elif '0.6' in self.root:
                    bias = 0.6
                elif '0.7' in self.root:
                    bias = 0.7
                elif '0.8' in self.root:
                    bias = 0.8
                elif '0.9' in self.root:
                    bias = 0.9
                possible_labels = [0, 1, 2] 
                probs = [0.33, 0.33, 1 - 0.66] 
                if self.mode == 'train':
                    base_num = np.random.choice([0, 1], p=[1 - bias, bias])
                    if base_num == 1:
                        x[:, :] = y
                    else:
                        possible_labels.pop(y)
                        base_num = np.random.choice(possible_labels, p=[0.5, 0.5])
                        x[:, :] = base_num
                else:
                    base_num = np.random.choice(possible_labels, p=probs)
                    x[:, :] = base_num
            else:
                x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            data = Data(x=x,
                        y=y,
                        z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=p,
                        edge_gt_att=torch.LongTensor(ground_truth),
                        name=f'SPMotif-{self.mode}-{idx}',
                        idx=idx)
            # add self loops
            # data.edge_index,data.edge_attr = add_self_loops(data.edge_index,data.edge_attr.squeeze(1))
            # data.edge_attr = data.edge_attr.unsqueeze(1)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])

    def add_self_loops(self):
        self.remove_self_loops()
        # print(torch.unique(self.data.edge_index).size())
        self.data.edge_index, self.data.edge_attr = add_self_loops(self.data.edge_index, self.data.edge_attr.squeeze(1))
        self.data.edge_attr = self.data.edge_attr.unsqueeze(1)

    def remove_self_loops(self):
        self.data.edge_index, self.data.edge_attr = remove_self_loops(self.data.edge_index, self.data.edge_attr)
