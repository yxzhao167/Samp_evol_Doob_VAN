import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from types import SimpleNamespace  
from scipy.signal import savgol_filter 
from gru import GRU
from gru2DFirstUp import GRU2D
from args import args
from SCGF import load_pre_trained_VAN_model_1D, load_pre_trained_VAN_model_2D,local_SCGF_1D,local_SCGF_2D,load_Doob_VAN

def occupy_density():
    args.c = 0.05
    args.Model = '2DSouthEast'
    args.net = 'rnn'
    args.L = 4
    args.size = args.L * args.L
    args.batch_size = 1000
    args.VAN_step = 3500
    args.VAN_step_next = 600
    args.net_depth = 3
    args.net_width = 32
    args.dlambda = 0.1
    args.cuda = 0
    args.dlambdaL = -2
    args.dlambdaR = 0.1
    args.device = 'cuda'
    args.lr = 10e-4
    args.Hermitian = True
    args.load_Doob = 1
    args.lambda_tilt = -1
 
    gru_kwargs = {  
        'L': args.L,  
        'size': args.L * args.L,
        'net_depth': args.net_depth, 
        'net_width': args.net_width,  
        'bias': True,  
        'z2': None,   
        'res_block': False,  
        'x_hat_clip': None,  
        'epsilon': 0.0, 
        'device': args.device,  
        'reverse': reversed,
        'binomialP': None,
    }  
 
    model = GRU2D(**gru_kwargs) 
    model.to(args.device)  
    model_path = f'trained_VAN_model_-1.0_4_0.05_2DSouthEast.pth'  
    checkpoint = torch.load(model_path, map_location=args.device)  
    model.load_state_dict(checkpoint)  
 
    total_occupy_density = torch.zeros((args.L, args.L), device=args.device)
    occupy_density_count = 0 
    sum_occupy_density_list = []
    it_range = 20
 
    for it in range(it_range):
        with torch.no_grad():
            sample, x_hat = model.sample(args.batch_size)
            LogP_t = model.log_prob(sample).detach()
        
        sample_occupy = (sample + 1) / 2
        occupy_density = torch.zeros((args.L, args.L), device=args.device)
        P_t = torch.exp(LogP_t)
        sum_P_t = torch.sum(P_t)
        
        for i in range(sample_occupy.size(0)):
            sample_contribution = sample_occupy[i] * P_t[i]
            occupy_density += sample_contribution
            
        occupy_density /= sum_P_t   
        sum_occupy_density = torch.sum(occupy_density) / args.size
        total_occupy_density += occupy_density  
        occupy_density_count += 1  
        sum_occupy_density_list.append(sum_occupy_density)
 
    N_i = sum(sum_occupy_density_list) / len(sum_occupy_density_list)
    n_i = total_occupy_density / occupy_density_count
 
    print(f"N_i: {N_i}, n_i (shape: {n_i.shape}):\n{n_i}\n")
    
    results = [{
        'lambda_tilt': args.lambda_tilt,
        't': it_range-1,
        'N_i': N_i,
        'n_i': n_i.cpu().numpy()
    }]
    
    with open('results.txt', 'a') as f:
        for result in results:
            f.write(f"lambda_tilt: {result['lambda_tilt']}, t: {result['t']}, N_i: {result['N_i']}, n_i:\n")
            f.write(f"{result['n_i']}\n\n")

if __name__ == '__main__':
    occupy_density()
  