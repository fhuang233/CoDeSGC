import numpy as np
import scipy.sparse as sp
import torch
import random
import os
from torch_geometric.utils import is_undirected, to_undirected, degree
from torch_sparse import SparseTensor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def index_to_mask(index, size):
    #mask = torch.zeros(size, dtype=torch.bool)
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
    

def random_splits(data, num_classes, percls_trn=20, val_lb=500):
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        #index = index[torch.randperm(index.size(0))]
        index = index[torch.randperm(index.size(0), device=index.device)]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    
    return data



def get_adj(data):

    edge_index = data.edge_index
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    
    edge_weight = torch.ones(edge_index.shape[1])
    n_node = data.num_nodes
    
    deg = degree(edge_index[0], n_node)
    deg[deg < 0.5] += 1.0
    deg = torch.pow(deg, -0.5)
    val = deg[edge_index[0]] * edge_weight * deg[edge_index[1]]
    
    #indices = torch.from_numpy(
        #np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    #values = torch.from_numpy(sparse_mx.data)
    #shape = torch.Size(sparse_mx.shape)
    #return torch.sparse.FloatTensor(indices, values, shape)
    
    #ret = SparseTensor(row=edge_index[0],
                       #col=edge_index[1],
                       #value=val,
                       #sparse_sizes=(n_node, n_node)).coalesce()
    ret = torch.sparse.FloatTensor(edge_index, val, (n_node, n_node)).coalesce()
    
    return ret



def get_propmat(data):
    edges = data.edge_index
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])),
                       shape=(data.num_nodes, data.num_nodes),
                       dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = adj - sp.diags(adj.diagonal()) 
    
    #adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = normalize_adj(adj)
    
    Lap = sp.eye(adj.shape[0]) - adj
    Lap2 = sp.eye(adj.shape[0]) + adj
    #Lap2_inv = sp.linalg.inv(Lap2)
    
    adj_matrix = sparse_mx_to_torch_sparse_tensor(adj)
    Lap_matrix = sparse_mx_to_torch_sparse_tensor(Lap)
    Lap_matrix2 = sparse_mx_to_torch_sparse_tensor(Lap2)
    #Lap_matrix2_inv = sparse_mx_to_torch_sparse_tensor(Lap2_inv)
    #return adj_matrix
    #return adj_matrix, Lap_matrix, Lap_matrix2, Lap_matrix2_inv
    return adj_matrix, Lap_matrix, Lap_matrix2
    

def load_parameters(args):
    if args.Net_t == 'linear':
        from linear_best_params import best_params
    if args.Net_t == 'hybrid':
        from hybrid_best_params import best_params
    for key, value in best_params[args.dataset][args.CoDe_t].items():
        args.__dict__[key] = value
    return args




