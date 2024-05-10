import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from typing import Iterable

from layers import ReLULayer, DropoutLayer, CPGCL, TuckerGCL, TuckerGCL_V, TuckerGCL_L


class GCLSeq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist: Iterable[nn.Module]):
        super(GCLSeq, self).__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, x, adj, **kwargs):
        out = self.modlist[0](x, adj, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out, adj, **kwargs)
        return out


class CoDeSGCN(nn.Module):
    def __init__(self, args):
        super(CoDeSGCN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        
        if args.CoDe_t == 'CP':
            SpecGCL = CPGCL
        elif args.CoDe_t == 'Tucker_V':
            SpecGCL = TuckerGCL_V
         
        elif args.CoDe_t == 'Tucker':
            SpecGCL = TuckerGCL
        else:
            SpecGCL = TuckerGCL_L
                  
        if args.Net_t == 'linear':
            self.register_parameter('Lins', None)
            self.GCLs = GCLSeq([SpecGCL(args.feat_dim, args.num_classes, args)])
        elif args.Net_t == 'multiGCL':
            self.register_parameter('Lins', None)
            GC1 = SpecGCL(args.feat_dim, args.hid_dim, args)
            GC2 = SpecGCL(args.hid_dim, args.num_classes, args)
            self.GCLs = GCLSeq([GC1, ReLULayer(), DropoutLayer(p=args.dropout), GC2])
        elif args.Net_t == 'hybrid':
            self.Lins = nn.Sequential(nn.Linear(args.feat_dim, args.hid_dim), 
                                      nn.ReLU(), nn.Dropout(p=args.dropout_ln))
            self.GCLs = GCLSeq([SpecGCL(args.hid_dim, args.num_classes, args)])
        
        self.LL_params = self.Lins.parameters() if self.Lins is not None else []
        self.GCparams1 = []
        self.GCparams2 = []
        self.GCparams3 = []
        self.GCparams4 = []
        self.GCparams5 = []
          
        for m in self.GCLs.modlist:
            if isinstance(m, SpecGCL):
                self.GCparams1.extend(list(m.weight_C.parameters()))
                self.GCparams2.extend(list(m.weight_D.parameters()))
                #--weight_D
                if args.CoDe_t != 'Tucker_L':
                    self.GCparams3.extend(list(m.weight_P.parameters())) 
                self.GCparams4.extend(list(m.conv.parameters())) #--decom_D
                if args.CoDe_t == 'Tucker':
                    self.GCparams5.extend(list(m.lamb.parameters()))
                #if args.CoDe_t == 'CP':
                    #self.GCparams5.append(m.lamb)   #--lamb
                #else: 
                    #self.GCparams5.extend(list(m.lamb.parameters()))
    
    def forward(self, x, adj, **kwargs):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.Lins is not None:
            x = self.Lins(x)
        out = self.GCLs(x, adj, **kwargs)
        
        return F.log_softmax(out, dim=1)


class GPRGNN(nn.Module):
    def __init__(self, args):
        super(GPRGNN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.dprate = args.dprate
        
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(args.feat_dim, args.hid_dim))
        #self.fcs.append(nn.BatchNorm1d(args.hid_dim))
        self.fcs.append(nn.Linear(args.hid_dim, args.num_classes))
        #self.fcs.append(nn.BatchNorm1d(args.num_classes))
        self.conv = GPR_convlayer(args.num_nodes, args.hid_dim, args.num_classes, args.order, args.att_droprate, args.dprate, bias=False, Init=args.Init)
        self.param = list(self.fcs.parameters())
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.fcs[1](self.fcs[0](x)))
        x = F.relu(self.fcs[0](x))
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.fcs[-1](self.fcs[2](x))
        x = self.fcs[-1](x)
        x = F.dropout(x, self.dprate, training=self.training)
        x = self.conv(x, adj)
        return F.log_softmax(x, dim=1)

