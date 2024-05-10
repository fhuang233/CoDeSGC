import torch_geometric.transforms as T
from tqdm import tqdm
import seaborn as sns
import numpy as np
import torch
import optuna
from parms_setting import settings
from utils import set_seed, random_splits
from train_eval import run
from datasets import DataLoader
from model import CoDeSGCN, GPRGNN
import math


def work(args, Net, dataset, percls_trn, val_lb):
    data = dataset[0]
    adj = dataset.adj
    
    data, adj = data.to(device), adj.to(device)  
    results = []
    
    for RP in tqdm(range(args.runs)):
        args.seed=RP
        set_seed(args.seed)
        data = random_splits(data, args.num_classes, percls_trn, val_lb)
        model = Net(args)
        model, data, adj = model.to(args.device), data.to(args.device), adj.to(args.device)
        if args.poly_t == 'Bernstein':
            L2 = dataset.L2.to(args.device)
            #L2_inv = dataset.L2_inv.to(args.device)
            test_acc, best_val_acc, time_run = run(model, adj, data, args, order=args.order, L2=L2)
        elif args.poly_t == 'Jacobi':
            test_acc, best_val_acc, time_run = run(model, adj, data, args, a=args.Jacobi_a, b=args.Jacobi_b)
        else: 
            test_acc, best_val_acc, time_run = run(model, adj, data, args)
        results.append(best_val_acc)
        
        #print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')
    return np.average(results)


def search_hyper_params(trial: optuna.Trial):
    
    if args.Net_t == 'hybrid':
        args.lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.005, 0.01, 0.05])
        args.wd_ln = trial.suggest_categorical("wd_ln", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
        args.dropout_ln = trial.suggest_float("dropout_ln", 0.0, 0.9, step=0.1)
    args.conv_lr1 = trial.suggest_categorical("conv_lr1", [0.0005, 0.001, 0.005, 0.01, 0.05])
    args.wd1 = trial.suggest_categorical("wd1", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    
    
    args.dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
    args.dprate1 = trial.suggest_float("dprate1", 0.0, 0.9, step=0.1)
    
    #args.poly_t = trial.suggest_categorical("poly_t", ['power', 'Legendre', 'Chebyshev', 'ChebyshevII', 'Jacobi'])
    if args.poly_t == 'Jacobi':
        args.Jacobi_a = trial.suggest_float('Jacobi_a', -1.0, 2.0, step=0.25)
        args.Jacobi_b = trial.suggest_float('Jacobi_b', -0.5, 2.0, step=0.25)
    
    if args.CoDe_t != 'Tucker_L':
        args.conv_lr3 = trial.suggest_categorical("conv_lr3", [0.0005, 0.001, 0.005, 0.01, 0.05])
        args.wd3 = trial.suggest_categorical("wd3", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
        args.dprate2 = trial.suggest_float("dprate2", 0.0, 0.9, step=0.1)
    if args.decom_D:
        args.conv_lr4 = trial.suggest_categorical("conv_lr4", [0.0005, 0.001, 0.005, 0.01, 0.05])
        args.wd4 = trial.suggest_categorical("wd4", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
        #args.alpha = trial.suggest_float('alpha', 0.5, 2.0, step=0.5)
    if args.weight_D:
        args.conv_lr2 = trial.suggest_categorical("conv_lr2", [0.0005, 0.001, 0.005, 0.01, 0.05])
        args.wd2 = trial.suggest_categorical("wd2", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    '''if args.lamb or args.CoDe_t == 'Tucker':
        args.conv_lr5 = trial.suggest_categorical("conv_lr5", [0.0005, 0.001, 0.005, 0.01, 0.05])
        args.wd5 = trial.suggest_categorical("wd5", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])'''
    
    #args.CP_rank = trial.suggest_int('CP_rank', 2, 32, 2)
    #if args.CoDe_t == 'CP':
        #args.CP_rank = 2**trial.suggest_int('log_CP_rank', 2, 5, step=1)
    #if args.ppr_init:
    #args.beta = trial.suggest_int('beta', 0, 10, step=1)  
    
    if args.CoDe_t == 'Tucker':
        #args.R_D = 2**trial.suggest_int('log_R_D', 2, 5, step=1)
        args.dprate3 = trial.suggest_float("dprate3", 0.0, 0.9, step=0.1)
        args.conv_lr5 = trial.suggest_categorical("conv_lr5", [0.0005, 0.001, 0.005, 0.01, 0.05])
        args.wd5 = trial.suggest_categorical("wd5", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    
    if args.CoDe_t != 'CP':
        args.R_D = 2**trial.suggest_int('log_R_D', 2, 5, step=1)
    
    return work(args, Net, dataset, percls_trn, val_lb)


args = settings()
set_seed(args.seed)
if torch.cuda.is_available():
    device = torch.device('cuda', args.cuda)
else:
    device = torch.device('cpu')
args.device = device

dataset = DataLoader('./data/', args.dataset, transform=T.NormalizeFeatures())
print(dataset[0])

args.num_classes = dataset.num_classes
args.feat_dim = dataset.num_features
args.num_nodes = dataset.data.num_nodes 
if args.split_t == 'dense':
    train_rate = 0.6
    val_rate = 0.2
else:
    train_rate = 0.025
    val_rate = 0.025
percls_trn = int(round(train_rate*args.num_nodes/args.num_classes))
val_lb = int(round(val_rate*args.num_nodes))
#10 fixed seeds for random splits from BernNet
#SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

print(args)
print("---------------------------------------------")

gnn_name = args.model_name
if gnn_name == 'CoDeSGCN':
    Net = CoDeSGCN
elif gnn_name =='GPRGNN':
    Net = GPRGNN

max_CP_R = ((args.order+1)*args.num_classes*args.feat_dim)/((args.order+1)+args.num_classes+args.feat_dim)
max_CP_R = int(math.log(max_CP_R, 2))
max_R_D = (args.feat_dim*(args.num_classes*(args.order+1)-args.R_C)-args.num_classes*args.R_P)/((args.order+1)+args.R_C*args.R_P)
print(max_R_D)
max_R_D = int(math.log(max_R_D, 2))

study = optuna.create_study(direction="maximize", storage="sqlite:///" + args.dataset + ".db",
                            study_name=args.Net_t+'_'+args.poly_t+'_'+ args.CoDe_t+'_'+str(args.output), load_if_exists=True)
study.optimize(search_hyper_params, n_trials=args.optruns)
print("best params ", study.best_params)
print("best valf1 ", study.best_value)