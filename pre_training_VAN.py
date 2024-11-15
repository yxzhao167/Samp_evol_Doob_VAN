from utils import (
    clear_checkpoint,
    clear_log,
    default_dtype_torch,
    ensure_dir,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
)
from SCGF import pre_training_VAN_1D, pre_training_VAN_1D_next, pre_training_VAN_2D, pre_training_VAN_2D_next
from gru import GRU
from gru2DFirstUp import GRU2D
from args import args
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import time

def test_SCGF():

    #setting the system parameters
    args.c = 0.5  #flipping rate
    args.Model = '2DSouthEast'  #'1DFA' or '1DEast' or '2DFA' or '2DSouthEast'
    args.net = 'rnn'
    args.L = 4   # Lattice size
    args.batch_size = 1000    # 1000 #batch size
    args.pre_train_step = 3000 # iteration steps for the first lambda_tilt
    args.pre_train_step_next = 300 #iteration steps for train VAN for other lambda_tilt
    args.net_depth = 3 # Depth of the neural network
    args.net_width = 32 # Width of the neural network
    args.dlambda = 0.1 # The steplength from the left to the rigth boundary value of the count field  (lambda=s)
    args.dlambdaL = 0  # left boundary value of the count field  (lambda=s)
    args.dlambdaR = 0.1  # Rigth boundary value of the count field  (lambda=s)
    args.device = 'cuda'
    args.lr = 10e-4 #learning rate
    args.Hermitian = True
    lambda_tilt_Range = 10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda))
    count = -1

    #instantiating neural network 
    if args.Model == '1DFA' or args.Model == '1DEast':
        params = {  
        'L': args.L,                  
        'net_depth': args.net_depth,           
        'net_width': args.net_width,          
        'bias': True,             
        'z2': False,              
        'res_block': False,       
        'x_hat_clip': 0.0,        
        'epsilon': 1e-12,          
        'device': args.device  
        } 
        net = GRU(**params)
        net.to(net.device)
        args.size = args.L

    else:
        params = {  
        'L': args.L,  
        'size':args.L ** 2,
        'net_depth': args.net_depth, 
        'net_width': args.net_width,  
        'bias': True,  
        'z2': None,   
        'res_block': False,  
        'x_hat_clip': None,  
        'epsilon': 1e-12, 
        'device': args.device,  
        'reverse':reversed,
        'binomialP':None,
        }  
        net = GRU2D(**params)
        net.to(net.device)
        args.size = args.L ** 2

    for i in range(0, len(lambda_tilt_Range)):
        count += 1
        if i == 0:
            args.lambda_tilt = -lambda_tilt_Range[0]  
            print(f"s:{args.lambda_tilt}")
            if args.Model == '1DFA' or args.Model == '1DEast':
                SCGF_1D = pre_training_VAN_1D(args)
                torch.cuda.empty_cache()
            else:
                SCGF_2D = pre_training_VAN_2D(args)
                torch.cuda.empty_cache()
        else:
            args.lambda_tilt_prev = lambda_tilt_Range[i - 1]
            args.lambda_tilt = lambda_tilt_Range[i]
            print('s:',args.lambda_tilt)
            if args.Model == '1DFA' or args.Model == '1DEast':
                SCGF_1D = pre_training_VAN_1D_next(args)
                torch.cuda.empty_cache()
            else:
                SCGF_2D = pre_training_VAN_2D(args)
                torch.cuda.empty_cache()

        
if __name__ == '__main__':
    test_SCGF()