
import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import scipy.special
import torch
from numpy import sqrt
from torch import nn

from args import args
from gru2DFirstUp import GRU2D
from gru import GRU
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def local_SCGF_1D(sample, args, net_new):
    Sample1D = (sample.view(-1, args.size) + 1) / 2 #batchsize X systemSize
    Win = (Sample1D - 1).abs() * (1 - args.c) + Sample1D * args.c 
    if args.Model == '1DFA':
        fNeighbor = torch.cat(
            (torch.ones(Sample1D.shape[0], 1).to(args.device), Sample1D[:, :-1]), 1
        ) + torch.cat((Sample1D[:, 1:], torch.ones(Sample1D.shape[0], 1).to(args.device)), 1)
    if args.Model == '1DEast':
        fNeighbor = torch.cat((torch.ones(Sample1D.shape[0], 1).to(args.device), Sample1D[:, :-1]), 1)
    SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size).expand(Sample1D.shape[0], args.size, args.size).to(args.device)
    SampleNeighbor1D = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)
    R = torch.sum((1 - Win) * fNeighbor, 1)
    Win = (Win * fNeighbor)  
    if not isinstance(args.lambda_tilt, torch.Tensor):  
        args.lambda_tilt = torch.tensor(args.lambda_tilt, dtype=torch.float64, device=args.device)
    Win_lambda = Win * torch.exp(-args.lambda_tilt)  
    if args.Hermitian:
        WinHermitian = torch.sqrt(torch.tensor(args.c * (1 - args.c), dtype=torch.float64, device=args.device)) * (  
        (Sample1D - 1).abs() + Sample1D)   
        # The previous state flip into these sampled states
        # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
        WinHermitian = WinHermitian * fNeighbor
        Win_lambda = WinHermitian * torch.exp(-args.lambda_tilt)
    LogP_t = net_new.log_prob(sample)
    Temp = torch.transpose(SampleNeighbor1D, 0, 1).view(
                sample.shape[0], args.size, args.size)
    Temp = Temp + (Temp - 1)
    if args.net == 'rnn':
        Temp2 = torch.reshape(Temp, (args.batch_size * args.size, args.size))
    else:
        Temp2 = torch.reshape(Temp, (args.batch_size * args.size, 1, args.size))
    LogP_t_other = torch.reshape(net_new.log_prob(Temp2), (args.batch_size, args.size)).detach()
    LogP_t_expanded = LogP_t.unsqueeze(1).repeat(1, args.size)
    thetaLoc = (
        torch.sum(torch.sqrt(torch.exp(LogP_t_other - LogP_t_expanded)) * Win_lambda, 1)
        - R
    )

    return thetaLoc

def local_SCGF_2D(sample, args, step, net):
    Sample1D = (sample.view(-1, args.size) + 1) / 2  # sample has size batchsize X systemSize

    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size).expand(Sample1D.shape[0], args.size, args.size).to(args.device)
    SampleNeighbor1DExtend = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)
    Sample1D = Sample1D[:, 1:]  # L^3 to L^3-1 neighbor by fixing the first spin up

    Sample2D = (sample.view(-1, args.L, args.L) + 1) / 2  # sample has size batchsize X L X L
    Win = (Sample1D - 1).abs() * (
        1 - args.c
    ) + Sample1D * args.c  # The previous state flip into these sampled states
    Col1 = torch.cat(
        (torch.zeros(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
    )  # down sites
    Col2 = torch.cat((Sample2D[:, :, 1:], torch.zeros(Sample2D.shape[0], args.L, 1).to(args.device)), 2)
    Row1 = torch.cat((torch.zeros(Sample2D.shape[0], 1, args.L).to(args.device), Sample2D[:, :-1, :]), 1)
    Row2 = torch.cat((Sample2D[:, 1:, :], torch.zeros(Sample2D.shape[0], 1, args.L).to(args.device)), 1)
    if args.BC == 1:
        if step == 0:
            print('Left-up BC SCGF.')
        Col1 = torch.cat(
            (torch.ones(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
        )  # down sites
    if args.BC == 2:
        if step == 0:
            print('All-up BC SCGF.')
        Col1 = torch.cat(
            (torch.ones(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
        )  # down sites
        Col2 = torch.cat((Sample2D[:, :, 1:], torch.ones(Sample2D.shape[0], args.L, 1).to(args.device)), 2)
        Row1 = torch.cat((torch.ones(Sample2D.shape[0], 1, args.L).to(args.device), Sample2D[:, :-1, :]), 1)
        Row2 = torch.cat((Sample2D[:, 1:, :], torch.ones(Sample2D.shape[0], 1, args.L).to(args.device)), 1)
    if args.BC == 3:
        if step == 0:
            print('Periodic BC.')
        Col1 = torch.cat(
            (Sample2D[:, :, -1].view(Sample2D.shape[0], args.L, 1), Sample2D[:, :, :-1]), 2
        )  # down sites
        Col2 = torch.cat((Sample2D[:, :, 1:], Sample2D[:, :, 0].view(Sample2D.shape[0], args.L, 1)), 2)
        Row1 = torch.cat((Sample2D[:, -1, :].view(Sample2D.shape[0], 1, args.L), Sample2D[:, :-1, :]), 1)
        Row2 = torch.cat((Sample2D[:, 1:, :], Sample2D[:, 0, :].view(Sample2D.shape[0], 1, args.L)), 1)

    if args.Model == '2DFA':
        # torch.cat((torch.ones(Sample2D.shape[0],args.L,1),Sample2D[:,:-1,:]) ,2)+torch.cat((Sample1D[:,1:],torch.ones(Sample1D.shape[0],1)) ,1)
        fNeighbor = (Col1 + Col2 + Row1 + Row2).view(-1, args.size)
    if args.Model == '2DNoEast':
        # torch.cat((torch.ones(Sample2D.shape[0],args.L,1),Sample2D[:,:-1,:]) ,2)+torch.cat((Sample1D[:,1:],torch.ones(Sample1D.shape[0],1)) ,1)
        fNeighbor = (Col1 + Row1 + Row2).view(-1, args.size)
    if args.Model == '2DEast':
        fNeighbor = Col1.view(-1, args.size)
    if args.Model == '2DSouthEast':
        fNeighbor = (Col1 + Row1).view(-1, args.size)
    if args.Model == '2DNorthWest':
        fNeighbor = (Col2 + Row2).view(-1, args.size)
    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size - 1, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size - 1).expand(Sample1D.shape[0], args.size - 1, args.size - 1).to(args.device)
    SampleNeighbor1D = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)
    # BatchSize: The escape-probability for each sampled state to all connected states#torch.sum((Sample1D-1).abs()*args.c+Sample1D*(1-args.c),1)
    R = torch.as_tensor(torch.sum((1 - Win) * fNeighbor[:, 1:], 1), dtype=torch.float64).to(args.device)
    # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
    Win = Win * fNeighbor[:, 1:]

    Win_lambda = torch.as_tensor(Win * np.float64(np.exp(-args.lambda_tilt)), dtype=torch.float64).to(
        args.device
    )
    if args.Hermitian:
        WinHermitian = (Sample1D - 1).abs() * np.sqrt(args.c * (1 - args.c)) + Sample1D * np.sqrt(
            args.c * (1 - args.c)
        )  # The previous state flip into these sampled states
        # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
        WinHermitian = WinHermitian * fNeighbor[:, 1:]
        Win_lambda = torch.as_tensor(
            WinHermitian * np.float64(np.exp(-args.lambda_tilt)), dtype=torch.float64
        ).to(args.device)
    
    LogP_t = net.log_prob(sample)
    # .view(sample.shape[0], args.size, args.size) #BatchSize X NeighborSize X SystemSize
    Temp = torch.transpose(SampleNeighbor1DExtend, 0, 1)
    # Set the first spin up to avoid numerical problem when generating prob
    Temp[:, 0, 0] = torch.as_tensor(1).to(args.device, dtype=default_dtype_torch)

    Temp = Temp + (Temp - 1)  # Change 0 to -1 back
    if args.net == 'rnn' or args.net == 'rnn2' or args.net == 'lstm' or args.net == 'rnn3':
        Temp3 = torch.reshape(Temp, (args.batch_size * args.size, args.L, args.L))  # For RNN
    else:
        Temp3 = torch.reshape(Temp, (args.batch_size * args.size, 1, args.L, args.L))  # For VAN
    # BatchSize X NeighborSize: checked, it is consistent with for loop
    LogP_t_other = torch.reshape(net.log_prob(Temp3), (args.batch_size, args.size)).detach()
    LogP_t_other = LogP_t_other[:, 1:]  # fixing the first spin up

    #if args.load_Doob == 1:
        #log_psi_square,SCGF = load_VAN_model(sample,args)
        #log_psi_square_other_temp,SCGF_prime =load_VAN_model(Temp3, args)
        #log_psi_square_other = torch.reshape(log_psi_square_other_temp, (args.batch_size, args.size)).detach()  
        #psi_ratio = torch.exp((log_psi_square_other-log_psi_square.unsqueeze(1).repeat(1, args.size))/2)
        #Win_lambda = Win_lambda * psi_ratio[:, 1:] 

    thetaLoc = (
        torch.sum(torch.sqrt(torch.exp(LogP_t_other - LogP_t.repeat(args.size - 1, 1).t())) * Win_lambda, 1)
        - R
    )  # Conversion from probability P to state \psi

    return thetaLoc

def pre_training_VAN_1D(args):    
    # initial 
    # varitional Wavefunction and SCGF
    gru_kwargs = {  
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
    model = GRU(**gru_kwargs)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), args.lr)

    for it in range(args.pre_train_step):
        optimizer.zero_grad() 
        with torch.no_grad():
            sample,_ = model.sample(args.batch_size)
            local_theta = local_SCGF_1D(sample, args,model)
        SCGF_mean = torch.mean(local_theta).item()  
        SCGF_var = local_theta.var().item()  

        if it % 10 ==0:
            print('mean(SCGF): {0}, var(SCGF): {1}, #samples {2}, #Step {3} \n\n'.format(SCGF_mean, SCGF_var, args.batch_size, it))  

        log_probs = model.log_prob(sample).to(args.device)   
        loss_reinforce = -1 * torch.mean((local_theta - local_theta.mean()) * log_probs)
        
        loss_reinforce.backward()    
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()  


    model_state_dict = model.state_dict()  
    torch.save(model_state_dict, f'pre_trained_VAN_model_{args.lambda_tilt}_{args.L}_{args.c}_{args.Model}.pth')  
    print("Model and SCGF_mean have been saved.")

    return SCGF_mean

def pre_training_VAN_1D_next(args):
    gru_kwargs = {  
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
    model = GRU(**gru_kwargs) 
    model.to(args.device)  
    # Load the previously trained model  
    model_path = f'pre_trained_VAN_model_{args.lambda_tilt_prev}_{args.L}_{args.c}_{args.Model}.pth'  
    checkpoint = torch.load(model_path, map_location=args.device)  
    model.load_state_dict(checkpoint)    # Load pre-trained weights  
 
  
    # Define optimizer (you may want to adjust the learning rate or other parameters)  
    optimizer = optim.Adam(model.parameters(), args.lr)  
  
    # Continue training  
    for it in range(args.pre_train_step_next):
        optimizer.zero_grad() 
        with torch.no_grad():
            sample,_ = model.sample(args.batch_size)
            local_theta = local_SCGF_1D(sample, args,model)
        SCGF_mean = torch.mean(local_theta).item()  
        SCGF_var = local_theta.var().item()   
          

        if it % 10 == 0:  
            print('mean(SCGF): {0}, var(SCGF): {1}, #samples {2}, #Step {3} \n\n'.format(SCGF_mean, SCGF_var, args.batch_size, it))  
  
        log_probs = model.log_prob(sample).to(args.device)   
        loss_reinforce = -1 * torch.mean((local_theta - local_theta.mean()) * log_probs)
  
        loss_reinforce.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  
        optimizer.step()  
  

    # Save the newly trained model  
    model_state_dict = model.state_dict()    
    torch.save(model_state_dict, f'pre_trained_VAN_model_{args.lambda_tilt}_{args.L}_{args.c}_{args.Model}.pth' )  
    print("Newly trained model and SCGF_mean have been saved.")  
  
    return SCGF_mean


def load_pre_trained_VAN_model_1D(args):  
    gru_kwargs = {  
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
    model = GRU(**gru_kwargs)  
    model.to(args.device)  

    model_path = f'pre_trained_VAN_model_{args.lambda_tilt}_{args.L}_{args.c}_{args.Model}.pth'
    checkpoint = torch.load(model_path, map_location=args.device)  
    model.load_state_dict(checkpoint)   
    model.eval() 
      
    return model



def pre_training_VAN_2D(args):    
    # initial 
    #RNNWavefunction and varitional SCGF
    gru_kwargs = {  
        'L': args.L,  
        'size':args.L * args.L,
        'net_depth': args.net_depth, 
        'net_width': args.net_width,  
        'bias': True,  
        'z2': None,   
        'res_block': False,  
        'x_hat_clip': None,  
        'epsilon': 0.0, 
        'device': args.device,  
        'reverse':reversed,
        'binomialP':None,
    }  
    model = GRU2D(**gru_kwargs)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    max_local_theta = float('-inf')
    last_30_SCGF_means = []  

    for it in range(args.pre_train_step):
        optimizer.zero_grad() 
        with torch.no_grad():
            sample,_ = model.sample(args.batch_size)
            local_theta = local_SCGF_2D(sample, args, 0,model)
        #local_theta = local_theta.to(args.device)
        SCGF_mean = torch.mean(local_theta).item()  
        SCGF_var = local_theta.var().item()  

        if it % 10 ==0:
            print('mean(SCGF): {0}, var(SCGF): {1}, #samples {2}, #Step {3} \n\n'.format(SCGF_mean, SCGF_var, args.batch_size, it))  

        log_probs = model.log_prob(sample).to(args.device)   
        loss_reinforce = -1 * torch.mean((local_theta - local_theta.mean()) * log_probs)
        
        loss_reinforce.backward()    
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()  


    #final_scgf_mean = sum(last_30_SCGF_means) / len(last_30_SCGF_means)  
    #print(f"Final SCGF Mean: {final_scgf_mean}") 
    model_state_dict = model.state_dict()  
    #model_state_dict['SCGF_mean'] = torch.tensor(final_scgf_mean)  
    torch.save(model_state_dict, f'pre_trained_VAN_model_{args.lambda_tilt}_{args.L}_{args.c}_{args.Model}.pth')  
    print("Model and SCGF_mean have been saved.")

    return 0

def pre_training_VAN_2D_next(args):
    gru_kwargs = {  
        'L': args.L,  
        'size':args.L * args.L,
        'net_depth': args.net_depth,  
        'net_width': args.net_width,  
        'bias': True,  
        'z2': None,   
        'res_block': False,  
        'x_hat_clip': None,  
        'epsilon': 0.0, 
        'device': args.device,  
        'reverse':reversed,
        'binomialP':None,
    }  
    model = GRU2D(**gru_kwargs) 
    model.to(args.device)  
    # Load the previously trained model  
    model_path = f'pre_trained_VAN_model_{args.lambda_tilt_prev}_{args.L}_{args.c}_{args.Model}.pth'  
    checkpoint = torch.load(model_path, map_location=args.device)  
    #__ = checkpoint.pop('SCGF_mean').item()
    model.load_state_dict(checkpoint)    # Load pre-trained weights  
    optimizer = optim.Adam(model.parameters(), args.lr)  
  

    
    # Continue training  
    for it in range(args.pre_train_step_next):  # This should be the new number of steps for the next training phase  
        optimizer.zero_grad()  
        with torch.no_grad():
            sample, _ = model.sample(args.batch_size)  
            local_theta = local_SCGF_2D(sample, args, 0, model)  
        #local_theta = local_theta.to(args.device)  
        SCGF_mean = torch.mean(local_theta).item()  
        SCGF_var = local_theta.var().item()  
          
        #if it >= args.VAN_step_next - 30:  
            #last_30_SCGF_means.append(SCGF_mean)  
          
        if it % 10 == 0:  
            print('mean(SCGF): {0}, var(SCGF): {1}, #samples {2}, #Step {3} \n\n'.format(SCGF_mean, SCGF_var, args.batch_size, it))  
  
        log_probs = model.log_prob(sample).to(args.device)   
        loss_reinforce = -1 * torch.mean((local_theta - local_theta.mean()) * log_probs)
  
        loss_reinforce.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  
        optimizer.step()  
  
    #SCGF_mean = sum(last_30_SCGF_means) / len(last_30_SCGF_means)  
    #print(f"Final SCGF Mean (next training): {SCGF_mean}")  
  
    # Save the newly trained model  
    model_state_dict = model.state_dict()  
    #model_state_dict['SCGF_mean'] = torch.tensor(SCGF_mean)  
    torch.save(model_state_dict, f'pre_trained_VAN_model_{args.lambda_tilt}_{args.L}_{args.c}_{args.Model}.pth' )  
    print("Newly pre_trained model and SCGF_mean have been saved.")  
  
    return 0


def load_pre_trained_VAN_model_2D(args):  
    gru_kwargs = {  
        'L': args.L,  
        'size':args.L * args.L,
        'net_depth': args.net_depth, 
        'net_width': args.net_width,  
        'bias': True,  
        'z2': None,   
        'res_block': False,  
        'x_hat_clip': None,  
        'epsilon': 0.0, 
        'device': args.device,  
        'reverse':reversed,
        'binomialP':None,
    }   
    model = GRU2D(**gru_kwargs) 
    model.to(args.device)  

    model_path = f'pre_trained_VAN_model_{args.lambda_tilt}_{args.L}_{args.c}_{args.Model}.pth'
    checkpoint = torch.load(model_path, map_location=args.device)  
    #SCGF_mean = checkpoint.pop('SCGF_mean').item()
    model.load_state_dict(checkpoint)   
    model.eval() 
      
    return model

def load_Doob_VAN(args):
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
        model = GRU(**params)
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
        model = GRU2D(**params)
        args.size = args.L ** 2
    
    model.to(args.device)  
    model_path = f'out{args.Tstep}{args.lambda_tilt}{args.L}{args.Model}.pth'
    checkpoint = torch.load(model_path, map_location=args.device)  
    model.load_state_dict(checkpoint)   
    model.eval() 
      
    return model
