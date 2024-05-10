#!/bin/sh

for dataset in cora citeseer pubmed computers photo chameleon actor squirrel texas cornell;
do
    for Net_t in hybrid linear;
    do
        for CoDe_t in Tucker_V Tucker_L Tucker CP;
        do
            python main.py --model_name CoDeSGCN --split_t dense --poly_t Jacobi --dataset=$dataset --Net_t=$Net_t --cuda 0 --runs 10 --CoDe_t=$CoDe_t --use_bias --weight_D --record
        done
    done
done
