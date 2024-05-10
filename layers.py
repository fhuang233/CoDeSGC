import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.special import comb
from PolyConv import Conv


class ReLULayer(nn.Module):
    def __init__(self):
        super(ReLULayer, self).__init__(inplace=False)
        self.relu = nn.ReLU(inplace=inplace)
    def forward(x, adj, **kwargs):
        return self.relu(x)


class DropoutLayer(nn.Module):
    def __init__(self):
        super(DropoutLayer, self).__init__(p=0.5, inplace=False)
        self.dropout = nn.Dropout(p, inplace)
    def forward(x, adj, **kwargs):
        return self.dropout(x)


class Combination(nn.Module):
    
    def __init__(self, args):
        super(Combination, self).__init__()
        self.order = args.order
        self.basebeta = args.beta
        
        if args.sole_D:
            self.weight = Parameter(torch.ones(1, args.order + 1, 1) / torch.arange(1, args.order + 2).unsqueeze(1),
                                    requires_grad=args.weight_D)
        else:
            self.weight = Parameter(torch.ones(1, args.order + 1, args.CP_rank) / torch.arange(1, args.order + 2).unsqueeze(1),
                                    requires_grad=args.weight_D)
        #self.weight = Parameter(torch.ones(1, args.order + 1, args.CP_rank),
                                #requires_grad=args.weight_D)
        if args.weight_D and not args.no_RandInit:
            self.reset_parameters()
        
    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
        #alpha = torch.rand(self.weight.size(2), 1)
        #alpha = torch.ones(self.weight.size(2), 1) * self.basebeta
        #self.weight.data = (alpha * (1. - alpha)**torch.arange(self.weight.size(1))).t().unsqueeze(0)
        #self.weight.data[0,-1,:] = (1. - alpha[:,0])**self.order
        
        #self.weight.data = self.weight.data * 0.0
        #self.weight.data[0, self.basebeta, :] = 1.0

    def forward(self, x):
        x = x * self.weight
        x = torch.sum(x, dim=1)
        return x

class CombinationII(nn.Module):
    
    def __init__(self, args):
        super(CombinationII, self).__init__()
        self.order = args.order
        self.basebeta = args.beta
        
        if args.sole_D:
            self.weight = Parameter(torch.ones(1, args.order + 1, 1, 1) / torch.arange(1, args.order + 2).unsqueeze(1).unsqueeze(2),
                                    requires_grad=args.weight_D)
        else:
            self.weight = Parameter(torch.ones(1, args.order + 1, 1, args.R_D) / torch.arange(1, args.order + 2).unsqueeze(1).unsqueeze(2),
                                    requires_grad=args.weight_D)
        
        if args.weight_D and not args.no_RandInit:
            self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
        #alpha = torch.rand(self.weight.size(3), 1)
        #alpha = torch.ones(self.weight.size(3), 1) * self.basebeta
        #self.weight.data = (alpha * (1. - alpha)**torch.arange(self.weight.size(1))).t().unsqueeze(0).unsqueeze(2)
        #self.weight.data[0,-1,0,:] = (1. - alpha[:,0])**self.order
        
        #self.weight.data = self.weight.data * 0.0
        #self.weight.data[0, self.basebeta, 0, :] = 1.0
        

    def forward(self, x):
        x = x * self.weight
        x = torch.sum(x, dim=3)
        x = torch.sum(x, dim=1)
        return x


class Poly_Conv(nn.Module):
    def __init__(self, args, rank):
        super(Poly_Conv, self).__init__()
        self.args = args
        self.order = args.order
        self.conv_t = args.poly_t
        self.conv_fn = Conv(args.poly_t)
        # 
        
        if args.decom_D:
            
            #self.weight = Parameter(torch.ones((args.order + 1, 1, rank)) * float(min(1 / args.alpha, 1)))
            self.weight = Parameter(torch.ones((args.order + 1, 1, rank)))
            #self.weight = Parameter(torch.FloatTensor(args.order + 1, 1, rank))
            #bound = 1. / math.sqrt(self.weight.size(0))
            # self.basealpha = (torch.rand(rank, 1)**torch.arange(args.order + 1)).t().unsqueeze(1).to(args.device)
            
            #self.basealpha = 2 * bound * self.basealpha - bound
            #self.basealpha = args.alpha
            #self.weight.data.uniform_(- 1.0 / bound, 1.0 / bound)
            #self.weight.data.uniform_(-0.5, 0.5)
            
            #self.weight = Parameter(torch.ones((args.order + 1, 1)) * float(min(1 / args.alpha, 1)))
            
        else:
            self.weight = Parameter(torch.ones((args.order + 1, 1, rank)), requires_grad=args.decom_D)
        
        #basealpha = torch.rand(args.order + 1, 1, rank) if args.decom_D else torch.ones((args.order + 1, 1, rank))
        #self.alphas = basealpha.to(args.device)
 
    def forward(self, inputs, adj=None, **kwargs):
        
        if self.args.decom_D:
 
            #alphas = self.basealpha * torch.tanh(self.weight)

            alphas = self.weight * torch.tanh(1.0 / (self.weight + 1e-5))
            
        else:
            alphas = self.weight
        #alphas = self.alphas
        x = inputs
        xs = []
        #if self.conv_t == 'Bernstein' or self.conv_t == 'BernsteinII':
        if self.conv_t == 'Bernstein':   
            L2 = kwargs['L2']
            temp_xs = [1. / (2**self.order) * x]
            for i in range(self.order):
                temp = torch.sparse.mm(L2, temp_xs[-1])
                temp_xs.append(temp)
            kwargs['temp_xs'] = temp_xs
            xs = [self.conv_fn(0, [temp_xs[-1]], adj, alphas, **kwargs)]
        else:
            xs = [self.conv_fn(0, [x], adj, alphas, **kwargs)]
        
        for i in range(1, self.order + 1):
            tmp_x = self.conv_fn(i, xs, adj, alphas, **kwargs)
            xs.append(tmp_x) 
        xs = [x.unsqueeze(1) for x in xs]
        x = torch.cat(xs, dim=1)  #signal_dim*order*rank
        return x


class CPGCL(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(CPGCL, self).__init__()
        self.args = args
        self.order = args.order
        self.dprate1 = args.dprate1
        self.dprate2 = args.dprate2
        self.conv_t = args.poly_t
        
        self.weight_C = nn.Linear(in_channels, args.CP_rank, bias=args.use_bias)
        self.weight_P = nn.Linear(args.CP_rank, out_channels, bias=args.use_bias)
        #self.lamb = Parameter(torch.ones(1, args.CP_rank), requires_grad=args.lamb)
        self.weight_D = Combination(args)
        self.conv = Poly_Conv(args, args.CP_rank)
        
        
    def forward(self, inputs, adj=None, **kwargs):
        
        x = self.weight_C(inputs)  
        x = F.dropout(x, self.dprate1, training=self.training)
        #x = x * self.lamb
        x = self.conv(x, adj, **kwargs)
        x = self.weight_D(x) 
        x = F.dropout(x, self.dprate2, training=self.training)
        x = self.weight_P(x)
        return x


class TuckerGCL(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(TuckerGCL, self).__init__()
        self.args = args
        self.order = args.order
        self.R_D = args.R_D
        self.R_P = args.R_P
        
        self.dprate1 = args.dprate1
        self.dprate2 = args.dprate2
        self.dprate3 = args.dprate3
        self.conv_t = args.poly_t
        
        self.weight_C = nn.Linear(in_channels, args.R_C, bias=args.use_bias)
        self.weight_P = nn.Linear(args.R_P, out_channels, bias=args.use_bias)
        #self.lamb = Parameter(torch.ones(args.R_C, args.R_D * args.R_P))
        self.lamb = nn.Linear(args.R_C, args.R_D * args.R_P, bias=args.use_bias)
        self.weight_D = CombinationII(args)
        self.conv = Poly_Conv(args, args.R_D * args.R_P)
        
    
    def forward(self, inputs, adj=None, **kwargs):
        
        x = self.weight_C(inputs)  
        x = F.dropout(x, self.dprate1, training=self.training)
        #x = torch.mm(x, self.lamb)
        x = self.lamb(x)
        x = F.dropout(x, self.dprate3, training=self.training)
        x = self.conv(x, adj, **kwargs).reshape(-1, self.order + 1, self.R_P, self.R_D)
        x = self.weight_D(x) 
        x = F.dropout(x, self.dprate2, training=self.training)
        x = self.weight_P(x)
        return x


class TuckerGCL_V(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(TuckerGCL_V, self).__init__()
        self.args = args
        self.order = args.order
        self.R_D = args.R_D
        self.R_P = args.R_P
        
        self.dprate1 = args.dprate1
        self.dprate2 = args.dprate2
        self.conv_t = args.poly_t
        
        self.weight_C = nn.Linear(in_channels, args.R_D * args.R_P, bias=args.use_bias)
        self.weight_P = nn.Linear(args.R_P, out_channels, bias=args.use_bias)
        #self.lamb = Parameter(torch.ones(args.R_C, args.R_D * args.R_P))
        #self.lamb = nn.Linear(args.R_C, args.R_D * args.R_P, bias=args.use_bias)
        self.weight_D = CombinationII(args)
        self.conv = Poly_Conv(args, args.R_D * args.R_P)
        
    
    def forward(self, inputs, adj=None, **kwargs):
        
        x = self.weight_C(inputs)  
        x = F.dropout(x, self.dprate1, training=self.training)
        #x = torch.mm(x, self.lamb)
        #x = self.lamb(x)
        #x = F.dropout(x, self.dprate3, training=self.training)
        x = self.conv(x, adj, **kwargs).reshape(-1, self.order + 1, self.R_P, self.R_D)
        x = self.weight_D(x) 
        x = F.dropout(x, self.dprate2, training=self.training)
        x = self.weight_P(x)
        return x

    
class TuckerGCL_L(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(TuckerGCL_L, self).__init__()
        self.args = args
        self.order = args.order
        self.R_D = args.R_D
        self.out_channels = out_channels
        
        self.dprate1 = args.dprate1
        self.dprate2 = args.dprate2
        self.conv_t = args.poly_t
        
        self.weight_C = nn.Linear(in_channels, args.R_D * out_channels, bias=args.use_bias)
        #self.weight_P = nn.Linear(args.R_P, out_channels, bias=args.use_bias)
        #self.lamb = Parameter(torch.ones(args.R_C, args.R_D * args.R_P))
        #self.lamb = nn.Linear(args.R_C, args.R_D * args.R_P, bias=args.use_bias)
        self.weight_D = CombinationII(args)
        self.conv = Poly_Conv(args, args.R_D * out_channels)
        
    
    def forward(self, inputs, adj=None, **kwargs):
        
        x = self.weight_C(inputs)  
        x = F.dropout(x, self.dprate1, training=self.training)
        #x = torch.mm(x, self.lamb)
        #x = self.lamb(x)
        #x = F.dropout(x, self.dprate3, training=self.training)
        x = self.conv(x, adj, **kwargs).reshape(-1, self.order + 1, self.out_channels, self.R_D)
        x = self.weight_D(x) 
        #x = F.dropout(x, self.dprate2, training=self.training)
        #x = self.weight_P(x)
        return x


