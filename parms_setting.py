import argparse


def settings():
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--is_cuda', action='store_true', default=False)
    # public parameters
    parser.add_argument('--model_name', type=str, default='CoDeSGCN')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split_t', type=str, default='dense')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--conv_lr1', type=float, default=0.0)
    parser.add_argument('--conv_lr2', type=float, default=0.0)
    parser.add_argument('--conv_lr3', type=float, default=0.0)
    parser.add_argument('--conv_lr4', type=float, default=0.01)
    parser.add_argument('--conv_lr5', type=float, default=0.01)
    
    parser.add_argument('--wd_ln', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd1', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd2', type=float, default=0.0, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd3', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd4', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd5', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed. Default is 0.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='gpu_id. Default is 0')
    
    parser.add_argument('--poly_t', type=str,
                        choices=['power', 'Legendre', 'Chebyshev', 'ChebyshevII', 'Jacobi', 'Bernstein'],
                        default='Jacobi') 
    parser.add_argument('--CoDe_t', type=str,
                        choices=['CP', 'Tucker', 'Tucker_V', 'Tucker_L'],
                        default='CP') 
       
    parser.add_argument('--Net_t', type=str,
                        choices=['linear', 'multiGCL', 'hybrid'],
                        default='linear') 
     
    parser.add_argument('--use_bias', action='store_true', default=False)
    parser.add_argument('--lamb', action='store_true', default=False)
    parser.add_argument('--decom_D', action='store_true', default=False)
    parser.add_argument('--no_RandInit', action='store_true', default=False)
    parser.add_argument('--weight_D', action='store_true', default=False)
    parser.add_argument('--sole_D', action='store_true', default=False)
    
    
    parser.add_argument('--hid_dim', type=int, default=64,
                        help='Number of hidden units for encoding layer. Default is 64.')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')
    parser.add_argument('--dropout_ln', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')
    parser.add_argument('--dprate1', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')
    parser.add_argument('--dprate2', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability). Default is 0.5.') 
    parser.add_argument('--dprate3', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')                   
    
    parser.add_argument('--Jacobi_a', type=float, default=0.0)
    parser.add_argument('--Jacobi_b', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=int, default=1.0)
    parser.add_argument('--order', type=int, default=10) 
    
    parser.add_argument('--CP_rank', type=int, default=32) 
    parser.add_argument('--R_C', type=int, default=32)
    parser.add_argument('--R_D', type=int, default=32)
    parser.add_argument('--R_P', type=int, default=32)
    
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--output', type=int, default='1')  
    parser.add_argument('--optruns', type=int, default=500)
   
    
    args = parser.parse_args()

    return args