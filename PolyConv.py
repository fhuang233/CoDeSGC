import torch
from scipy.special import comb
from torch_geometric.utils import add_self_loops, get_laplacian
from torch_geometric.nn import MessagePassing


def Conv(conv_t='power'):
    if conv_t == 'power': return PowerConv
    if conv_t == 'Legendre': return LegendreConv
    if conv_t == 'Chebyshev': return ChebyshevConv
    if conv_t == 'ChebyshevII': return ChebyshevConvII
    if conv_t == 'Jacobi': return JacobiConv
    if conv_t == 'Bernstein': return BernsteinConv
    #if conv_t == 'BernsteinII': return BernsteinConvII
'''
Multiple polynomial bases from the paper "How powerful are spectral graph neural networks"
'''

def PowerConv(k, xs, adj, alphas, **kwargs):
    '''
    Monomial bases.
    '''
    #if k == 0: return xs[0] * alphas[0]
    if k == 0: return xs[0]
    return  torch.sparse.mm(adj, xs[-1]) * alphas[k]
    #adj @ xs[-1]


def LegendreConv(k, xs, adj, alphas, **kwargs):
    '''
    Legendre bases. 
    '''
    #if k == 0: return xs[0] * alphas[0] 
    if k == 0: return xs[0]   
    #nx = (2 - 1 / k) * torch.sparse.mm(adj, xs[-1]) * alphas[k]
    nx = (2 - 1 / k) * torch.sparse.mm(adj, xs[-1]) * alphas[k - 1]
    if k > 1:
        # nx -= (1 - 1 / k) * xs[-2] * alphas[k - 1] * alphas[k]
        nx -= (1 - 1 / k) * xs[-2] * alphas[k - 2] * alphas[k - 1]
    return nx


def ChebyshevConv(k, xs, adj, alphas, **kwargs):
    '''
    Chebyshev Bases. first kind 
    '''
    #if k == 0: return xs[0] * alphas[0]
    if k == 0: return xs[0]
    #nx = torch.sparse.mm(adj, xs[-1]) * alphas[k]
    nx = torch.sparse.mm(adj, xs[-1]) * alphas[k - 1]
    #adj @ xs[-1]
    if k > 1:
        # nx = 2. * nx - xs[-2] * alphas[k - 1] * alphas[k]
        nx = 2. * nx - xs[-2] * alphas[k - 2] * alphas[k - 1]
    return nx


def ChebyshevConvII(k, xs, adj, alphas, **kwargs):
    '''
    Chebyshev Bases. second kind 
    '''
    if k == 0: return xs[0] * alphas[0]
    #if k == 0: return xs[0]
    # nx = 2. * torch.sparse.mm(adj, xs[-1]) * alphas[k]
    nx = 2. * torch.sparse.mm(adj, xs[-1]) * alphas[k - 1]
    if k > 1:
        # nx -= xs[-2] * alphas[k - 1] * alphas[k]
        nx -= xs[-2] * alphas[k - 2] * alphas[k - 1]
    return nx


def JacobiConv(k, xs, adj, alphas, **kwargs):
    '''
    Jacobi Bases. 
    '''
    a = kwargs['a']
    b = kwargs['b']
    #if k == 0: return xs[0] * alphas[0]
    if k == 0: return xs[0]
    if k == 1:
        coef1 = (a - b) / 2
        coef2 = (a + b + 2) / 2
        # return coef1 * xs[-1] * alphas[k] + coef2 * torch.sparse.mm(adj, xs[-1]) * alphas[k]
        return coef1 * xs[-1] * alphas[0] + coef2 * torch.sparse.mm(adj, xs[-1]) * alphas[0]
    coef_l = 2 * k * (k + a + b) * (2 * k - 2 + a + b)
    coef_lm1_1 = (2 * k + a + b - 1) * (2 * k + a + b) * (2 * k + a + b - 2)
    coef_lm1_2 = (2 * k + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (k - 1 + a) * (k - 1 + b) * (2 * k + a + b)
    tmp1 = coef_lm1_1 / coef_l
    tmp2 = coef_lm1_2 / coef_l
    tmp3 = coef_lm2 / coef_l
    # nx = tmp1 * torch.sparse.mm(adj, xs[-1]) * alphas[k] + tmp2 * xs[-1] * alphas[k]
    nx = tmp1 * torch.sparse.mm(adj, xs[-1]) * alphas[k - 1] + tmp2 * xs[-1] * alphas[k - 1]
    #nx -= tmp3 * xs[-2] * alphas[k - 1] * alphas[k]
    nx -= tmp3 * xs[-2] * alphas[k - 2] * alphas[k - 1]
    return nx


def BernsteinConv(k, xs, adj, alphas, **kwargs):
    '''
    Bernstein Bases, adj = I - adj
    '''
    if 'order' in kwargs.keys():
        K = kwargs['order']
    if 'temp_xs' in kwargs.keys():
        temp_xs = kwargs['temp_xs']
    #if k == 0: return xs[0] * alphas[0]
    if k == 0: return xs[0]
    # x =  temp_xs[K - k] * alphas[0]
    x =  temp_xs[K - k]
    for i in range(k):
        coef = (K - i) / (i + 1)
        x = coef * torch.sparse.mm(adj, x) * alphas[i + 1]
    return x
    
'''
def BernsteinConvII(k, xs, adj, alphas, **kwargs):

    # Bernstein Bases, adj = I - adj
    
    K = kwargs['order']
    # the inversion of I + adj
    L2_inv = kwargs['L2_inv']
    if k == 0: return xs[0] * alphas[0] 
    coef = (K - k + 1) / k
    nx = torch.sparse.mm(adj, xs[-1]) * alphas[k]
    return coef * torch.sparse.mm(L2_inv, nx) 
'''
    
    
class Bern_prop(MessagePassing):
    # Bernstein polynomial filter from the `"BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation" paper.
    # Copied from the official implementation.
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index,
                                           edge_weight,
                                           normalization='sym',
                                           dtype=x.dtype,
                                           num_nodes=x.size(0))
        #2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,
                                            -norm1,
                                            fill_value=2.,
                                            num_nodes=x.size(0))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = [(comb(self.K, 0) / (2**self.K)) * tmp[self.K]]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out.append((comb(self.K, i + 1) / (2**self.K)) * x)
        return  torch.stack(out, dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)