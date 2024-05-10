import torch_geometric.transforms as T
from tqdm import tqdm
import seaborn as sns
import numpy as np
import torch

from parms_setting import settings
from utils import set_seed, random_splits, load_parameters
from train_eval import run
from datasets import DataLoader
from model import CoDeSGCN, GPRGNN


args = settings()
args = load_parameters(args)

set_seed(args.seed)
if torch.cuda.is_available():
    device = torch.device('cuda', args.cuda)
else:
    device = torch.device('cpu')
args.device = device

dataset = DataLoader('./data/', args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]
adj = dataset.adj
if args.poly_t == 'Bernstein':
    adj = dataset.L

data, adj = data.to(device), adj.to(device)
    
args.num_classes = dataset.num_classes
args.feat_dim = dataset.num_features
args.num_nodes = data.num_nodes 
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
elif gnn_name == 'ChebNet':
    Net = ChebNet
elif gnn_name =='ChebBase':
    Net = ChebBase
elif gnn_name == "ChebNetII":
    Net = ChebNetII
    
results = []
time_results=[]

for RP in tqdm(range(args.runs)):
    args.seed=RP
    print(RP)
    set_seed(args.seed)
    
    data = random_splits(data, args.num_classes, percls_trn, val_lb)

    model = Net(args)
    model, data, adj = model.to(device), data.to(device), adj.to(device)
    if args.poly_t == 'Bernstein':
        L2 = dataset.L2.to(device)
        #L2_inv = dataset.L2_inv.to(device)
        test_acc, best_val_acc, time_run = run(model, adj, data, args, order=args.order, L2=L2)
    elif args.poly_t == 'Jacobi':
        test_acc, best_val_acc, time_run = run(model, adj, data, args, a=args.Jacobi_a, b=args.Jacobi_b)
    else: 
        test_acc, best_val_acc, time_run = run(model, adj, data, args)
    time_results.append(time_run)
    results.append([test_acc, best_val_acc])
    print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f} \t val_acc: {best_val_acc:.4f}')
        #if args.model_name in ["ChebBase","ChebNetII"]:
            #print('Weights:', [float('{:.4f}'.format(i)) for i in theta_0])

run_sum=0
epochsss=0
for i in time_results:
    run_sum+=sum(i)
    epochsss+=len(i)
print("each run avg_time:",run_sum/(args.runs),"s")
print("each epoch avg_time:",1000*run_sum/epochsss,"ms")
test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
values=np.asarray(results,dtype=object)[:,0]
uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))
print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
print(f'test acc mean={test_acc_mean:.4f}+/-{uncertainty*100:.4f}  \t val acc mean={val_acc_mean:.4f}')
print('val_acc_three_times_mean:', np.mean(results[:3], axis=0)[1] * 100)
if args.record:
    with open(args.dataset + '_' + args.Net_t + '_' + args.CoDe_t +'_result.txt', 'a') as f:
        f.write(f'Test Accuracy: {test_acc_mean:.4f} +/- {uncertainty*100:.4f}\n')
    