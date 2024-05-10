from typing import Optional, Callable, List, Union
import torch
import numpy as np
import pickle
import os.path as osp
import os
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_planetoid_data, read_npz
#from torch_geometric.utils import coalesce
from torch_sparse import coalesce
from utils import get_propmat, get_adj


class DataLoader(InMemoryDataset):

    def __init__(self, root: str, name: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'actor', 'squirrel', 
                             'computers', 'photo', 'texas', 'cornell']
        
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.adj = torch.load(osp.join(self.processed_dir, 'adj.pt'))
        self.L = torch.load(osp.join(self.processed_dir, 'L.pt'))
        self.L2 = torch.load(osp.join(self.processed_dir, 'L2.pt'))
        #self.L2_inv = torch.load(osp.join(self.processed_dir, 'L2_inv.pt'))


    @property
    def raw_dir(self) -> str:
        name = 'film' if self.name == 'actor' else self.name
        return osp.join(self.root, name, 'raw')

    @property
    def processed_dir(self) -> str:
        name = 'film' if self.name == 'actor' else self.name
        return osp.join(self.root, name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        if self.name in ['cora', 'citeseer', 'pubmed']:
            names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
            return [f'ind.{self.name.lower()}.{name}' for name in names]
        elif self.name in ['computers', 'photo']:
            return f'amazon_electronics_{self.name.lower()}.npz'
        
        elif self.name in ['chameleon', 'squirrel', 'texas', 'actor', 'cornell']:
            return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']
        
    #@property
    #def processed_paths(self) -> List[str]:
        #return [osp.join(self.processed_dir, f) for f in self.processed_file_names]
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'   
           
    def download(self):
        pass

    def process(self):
        
        if self.name in ['cora', 'citeseer', 'pubmed']:
            data = read_planetoid_data(self.raw_dir, self.name)
        elif self.name in ['computers', 'photo']:
            data = read_npz(self.raw_paths[0])
        
        elif self.name in ['actor', 'texas', 'cornell', 'chameleon', 'squirrel']:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
                if self.name == 'actor':
                    data = [x.split('\t') for x in data]
                    rows, cols = [], []
                        
                    for n_id, col, _ in data:
                        col = [int(x) for x in col.split(',')]
                        rows += [int(n_id)] * len(col)
                        cols += col
                    row, col = torch.tensor(rows), torch.tensor(cols)
    
                    x = torch.zeros(int(row.max()) + 1, int(col.max()) + 1)
                    x[row, col] = 1.
    
                    y = torch.empty(len(data), dtype=torch.long)
                    for n_id, _, label in data:
                        y[int(n_id)] = int(label)
                else:
                    x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
                    x = torch.tensor(x, dtype=torch.float)
    
                    y = [int(r.split('\t')[2]) for r in data]
                    y = torch.tensor(y, dtype=torch.long)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
                edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
                #edge_index = coalesce(edge_index, num_nodes=x.size(0))
                
                edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

            data = Data(x=x, edge_index=edge_index, y=y)
        
        data = data if self.pre_transform is None else self.pre_transform(data)
        #adj, L, L2, L2_inv = get_propmat(data)
        #adj = get_adj(data)
        #adj = get_propmat(data)
        adj, L, L2 = get_propmat(data)
        
        torch.save(adj, osp.join(self.processed_dir, 'adj.pt'))
        torch.save(L, osp.join(self.processed_dir, 'L.pt'))
        torch.save(L2, osp.join(self.processed_dir, 'L2.pt'))
        #torch.save(L2_inv, osp.join(self.processed_dir, 'L2_inv.pt'))
        torch.save(self.collate([data]), self.processed_paths[0])
        

    def __repr__(self) -> str:
        return f'{self.name}()'



