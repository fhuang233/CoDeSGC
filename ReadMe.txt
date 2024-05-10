Requirement: pytorch1.9.0+cu111; pytorch_geometric 1.7.2

For all results in Table 3, 'bash run.sh'.

For tunining the hyperparameters, For example: 'python hyper_opt.py --model_name CoDeSGCN --split_t dense --poly_t Jacobi --dataset computers --Net_t hybrid --cuda 1 --runs 3 --optruns 400 --output 0 --CoDe_t Tucker_V --use_bias --weight_D '

Tucker_V is the Tucker2 decomposition and Tucker_L is the Tucker1 decomposition.