import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
import numpy as np
import copy


def run(model, adj, data, args, **kwargs):
    
    if args.model_name=='CoDeSGCN':
        optimizer = Adam([{'params':model.LL_params,'weight_decay':args.wd_ln, 'lr': args.lr},
                        {'params':model.GCparams1,'weight_decay':args.wd1, 'lr': args.conv_lr1},
                        {'params':model.GCparams2,'weight_decay':args.wd2, 'lr': args.conv_lr2},
                        {'params':model.GCparams3,'weight_decay':args.wd3, 'lr': args.conv_lr3},
                        {'params':model.GCparams4,'weight_decay':args.wd4, 'lr': args.conv_lr4},
                        {'params':model.GCparams5,'weight_decay':args.wd5, 'lr': args.conv_lr5}
                        ])
        '''optimizer = Adam([{'params':model.LL_params,'weight_decay':args.wd_ln, 'lr': args.lr},
                        {'params':model.GCparams1,'weight_decay':args.wd1, 'lr': args.conv_lr1},
                        {'params':model.GCparams2,'weight_decay':args.wd2, 'lr': args.conv_lr2},
                        {'params':model.GCparams3,'weight_decay':args.wd3, 'lr': args.conv_lr3}
                        ])'''
    elif args.model_name=='GPRGNN':
        optimizer = torch.optim.Adam([{ 'params': model.param, 'weight_decay': args.wd1, 'lr': args.lr},
                                      {'params': model.conv.parameters(), 'weight_decay': args.wd2, 'lr': args.conv_lr}])
    
    #elif args.net =='BernNet':
        #optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        #{'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        #{'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    #else:
        #optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    best_val_acc = 0
    test_acc = 0
    early_stop = 0
    best_epoch = 0
    #best_alphas = 0
    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data, adj, args, **kwargs)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        eval_info = evaluate(model, data, adj, args, **kwargs)
        
        train_loss, val_loss, tmp_test_loss = eval_info['train_loss'], eval_info['val_loss'], eval_info['test_loss']
        train_acc, val_acc, tmp_test_acc = eval_info['train_acc'], eval_info['val_acc'], eval_info['test_acc']
        '''print('Epoch: {:04d}'.format(epoch),
                  '***train_loss: {:.4f}'.format(train_loss),
                  '***train_acc: {:.4f}'.format(train_acc), 
                  '***val_loss: {:.4f}'.format(val_loss),
                  '***val_acc: {:.4f}'.format(val_acc),
                  '***test_loss: {:.4f}'.format(tmp_test_loss),
                  '***test_acc: {:.4f}'.format(tmp_test_acc),
                  )'''
        
        if val_acc >= best_val_acc:
            early_stop = 0
            best_val_acc = val_acc
            best_epoch = epoch
            test_acc = tmp_test_acc
            #best_alphas = alphas
        else:
            early_stop += 1
        if early_stop > args.early_stopping:
            print('best epoch:', best_epoch)
            #print(best_alphas)
            break
    
    '''
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(1, args.epochs+1):
        t_st=time.time()
        train(model, optimizer, data, adj, args, **kwargs)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        eval_info = evaluate(model, data, adj, args, **kwargs)
        #[train_acc, val_acc, tmp_test_acc], preds, [
            #train_loss, val_loss, tmp_test_loss] = test(model, data)
        
        train_loss, val_loss, tmp_test_loss = eval_info['train_loss'], eval_info['val_loss'], eval_info['test_loss']
        train_acc, val_acc, tmp_test_acc = eval_info['train_acc'], eval_info['val_acc'], eval_info['test_acc']
        print('Epoch: {:04d}'.format(epoch),
                  '***train_loss: {:.4f}'.format(train_loss),
                  '***train_acc: {:.4f}'.format(train_acc), 
                  '***val_loss: {:.4f}'.format(val_loss),
                  '***val_acc: {:.4f}'.format(val_acc))
             
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            #if args.net =='BernNet':
                #TEST = tmp_net.prop1.temp.clone()
                #theta = TEST.detach().cpu()
                #theta = torch.relu(theta).numpy()
            #else:
                #theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    break'''
    '''             
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    early_stop = 0
    time_run=[]
    for epoch in range(1, args.epochs+1):
        t_st=time.time()
        train(model, optimizer, data, adj, args, **kwargs)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        eval_info = evaluate(model, data, adj, args, **kwargs)
        #[train_acc, val_acc, tmp_test_acc], preds, [
            #train_loss, val_loss, tmp_test_loss] = test(model, data)
        
        train_loss, val_loss, tmp_test_loss = eval_info['train_loss'], eval_info['val_loss'], eval_info['test_loss']
        train_acc, val_acc, tmp_test_acc = eval_info['train_acc'], eval_info['val_acc'], eval_info['test_acc']
        print('Epoch: {:04d}'.format(epoch),
                  '***train_loss: {:.4f}'.format(train_loss),
                  '***train_acc: {:.4f}'.format(train_acc), 
                  '***val_loss: {:.4f}'.format(val_loss),
                  '***val_acc: {:.4f}'.format(val_acc))
             
        if val_loss < best_val_loss:
            early_stop = 0
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            #if args.net =='BernNet':
                #TEST = tmp_net.prop1.temp.clone()
                #theta = TEST.detach().cpu()
                #theta = torch.relu(theta).numpy()
            #else:
                #theta = args.alpha
        else:
            early_stop += 1
        if early_stop > args.early_stopping:
            break'''
    
    '''
    best_val_acc = 0
    test_acc = 0
    best_val_loss = float('inf')
    early_stop = 0
    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data, adj, args, **kwargs)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)
        eval_info = evaluate(model, data, adj, args, **kwargs)
        
        train_loss, val_loss, tmp_test_loss = eval_info['train_loss'], eval_info['val_loss'], eval_info['test_loss']
        train_acc, val_acc, tmp_test_acc = eval_info['train_acc'], eval_info['val_acc'], eval_info['test_acc']
        print('Epoch: {:04d}'.format(epoch),
                  '***train_loss: {:.4f}'.format(train_loss),
                  '***train_acc: {:.4f}'.format(train_acc), 
                  '***val_loss: {:.4f}'.format(val_loss),
                  '***val_acc: {:.4f}'.format(val_acc))
        
        if val_acc >= best_val_acc:
            if val_loss <= best_val_loss:
                early_stop = 0
                best_val_acc = val_acc
                best_val_loss = val_loss
                test_acc = tmp_test_acc
        else:
            early_stop += 1
        if early_stop > args.early_stopping:
            break'''
                    
    return test_acc, best_val_acc, time_run
    

def train(model, optimizer, data, adj, args, **kwargs):
    model.train()
    
    optimizer.zero_grad()
    out = model(data.x, adj, **kwargs)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
    loss.backward()
    optimizer.step()
    #return alphas
    
 
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    
   
def evaluate(model, data, adj, args, **kwargs):
    model.eval()

    with torch.no_grad():
        logits = model(data.x, adj, **kwargs)
       
    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
        outs['{}_pred'.format(key)] = pred.detach().cpu()
 
    return outs
