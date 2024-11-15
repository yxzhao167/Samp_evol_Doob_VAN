import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from types import SimpleNamespace  
from scipy.signal import savgol_filter 
from gru import GRU
from gru2DFirstUp import GRU2D
from args import args
from SCGF import load_pre_trained_VAN_model_1D, load_pre_trained_VAN_model_2D,local_SCGF_1D,local_SCGF_2D,load_Doob_VAN

args.net_depth = 3
args.net_width = 32
args.c = 0.5
args.Model = '1DEast'
args.net = 'rnn'
args.L = 20
args.batch_size = 1000
args.VAN_step = 8000
args.dlambda = 0.1
args.cuda =0
args.dlambdaL = -3
args.dlambdaR = 0.1
lambda_tilt_Range = 10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda))
args.decive = 'cuda'
SCGF_means = []  
args.s_init = 0

with open('results.txt', 'w') as file:
    for i in range(0, 5):
        theta_mean = [] 
        args.lambda_tilt = lambda_tilt = lambda_tilt_Range[i]

        if args.Model == '1DFA' or args.Model == '1DEast':
            args.size = args.L 
            # obtain result from VMC
            #model = load_pre_trained_VAN_model_1D(args)
            # obtain result from Doob
            model = load_Doob_VAN(args)
        else:
            args.size = args.L * args.L
            #model = load_pre_trained_VAN_model_2D(args)
            model = load_Doob_VAN(args)
         
        # obtain result from Doob
            model.to(args.device) 
        for j in range(50):
            with torch.no_grad():
                sample, x_hat = model.sample(args.batch_size)
 
                if args.Model == '1DFA' or args.Model == '1DEast':
                    local_theta = local_SCGF_1D(sample, args, model)
                else:
                    local_theta = local_SCGF_2D(sample, args, 1, model)
            local_theta_mean = torch.mean(local_theta).item()  
            theta_mean.append(local_theta_mean)   
        theta = sum(theta_mean) / len(theta_mean)  
        theta = theta / args.size
        print(f"theta: {theta}") 

        file.write(f"{lambda_tilt} {theta} {args.L}\n")