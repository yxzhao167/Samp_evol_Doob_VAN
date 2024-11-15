# Transition rule for 2D KCM

import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
from numpy import sqrt
from torch import nn
from args import args
from gru import GRU
from gru2DFirstUp import GRU2D
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


plt.rc('font', size=16)

def TransitionState1D(sample, args, Tstep, net_new, pre_trained_model,theta):
    Sample1D = (sample.view(-1, args.size) + 1) / 2  # sample has size batchsize X systemSize
    Win = (Sample1D - 1).abs() * (
        1 - args.c
    ) + Sample1D * args.c  # The previous state flip into these sampled states
    if args.Model == '1DFA':
        fNeighbor = torch.cat(
            (torch.ones(Sample1D.shape[0], 1).to(args.device), Sample1D[:, :-1]), 1
        ) + torch.cat((Sample1D[:, 1:], torch.ones(Sample1D.shape[0], 1).to(args.device)), 1)
    if args.Model == '1DEast':
        fNeighbor = torch.cat((torch.ones(Sample1D.shape[0], 1).to(args.device), Sample1D[:, :-1]), 1)
    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size).expand(Sample1D.shape[0], args.size, args.size).to(args.device)
    SampleNeighbor1D = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)

    # BatchSize: The escape-probability for each sampled state to all connected states#torch.sum((Sample1D-1).abs()*args.c+Sample1D*(1-args.c),1)
    R = torch.sum((1 - Win) * fNeighbor, 1)
    Win = (
        Win * fNeighbor
    )  # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
    args.lambda_tilt = torch.tensor(args.lambda_tilt, dtype=torch.float64).to(args.device)
    Win_lambda = torch.tensor(Win * torch.exp(-args.lambda_tilt)).to(args.device)
    # Extract the elements for  the previous-step state:  for the samples, we need to get the probility in the state vector:
    # For initial steady-state, just count 1s 0s. For later steps, use index to sample for VAN
    if args.Hermitian:
        WinHermitian = (Sample1D - 1).abs() * np.sqrt(args.c * (1 - args.c)) + Sample1D * np.sqrt(
            args.c * (1 - args.c)
        )  # The previous state flip into these sampled states
        # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
        WinHermitian = WinHermitian * fNeighbor
        Win_lambda = torch.tensor(
            WinHermitian * np.float64(torch.exp(-args.lambda_tilt)), dtype=torch.float64
        ).to(args.device)

    if Tstep == 0:
        with torch.no_grad():
            Temp = torch.transpose(SampleNeighbor1D, 0, 1).view(
                sample.shape[0], args.size, args.size
            )  # BatchSize X NeighborSize X SystemSize
            Temp = Temp + (Temp - 1)  # Change 0 to -1 back
            Temp2 = torch.reshape(Temp, (args.batch_size * args.size, args.size))  #For networks

            ones = torch.sum((Sample1D == 1.0), 1)
            # BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
            P_t = args.c**ones * (1 - args.c) ** (args.size - ones)
            ones = (SampleNeighbor1D == 1.0).sum(dim=2)
            P_t_other = (args.c**ones * (1 - args.c) ** (args.size - ones)).t()  # BatchSize X NeighborSize
    else:
        with torch.no_grad():
            P_t = torch.exp(net_new.log_prob(sample)).detach()
            # Temp=torch.transpose(SampleNeighbor1D, 0, 1).view(args.batch_size, args.size, args.size) #BatchSize X NeighborSize X SystemSize
            Temp = torch.transpose(SampleNeighbor1D, 0, 1).view(
                sample.shape[0], args.size, args.size
            )  # BatchSize X NeighborSize X SystemSize
            Temp = Temp + (Temp - 1)  # Change 0 to -1 back

            Temp2 = torch.reshape(Temp, (args.batch_size * args.size, args.size))  #For networks

            P_t_otherTemp = torch.exp(net_new.log_prob(Temp2)).detach()
            P_t_other = torch.reshape(P_t_otherTemp, (args.batch_size, args.size))
            # P_t_other=torch.exp(net_new.log_prob2(Temp)).detach()#BatchSize X NeighborSize: checked, it is consistent with for loop

    if args.Doob == 1:
            log_psi = pre_trained_model.log_psi(sample.to(args.device)).detach() 
            log_psi_other_temp = pre_trained_model.log_psi(Temp2.to(args.device)).detach()  
            log_psi_other = torch.reshape(log_psi_other_temp, (args.batch_size, args.size)).detach()
            psi_ratio = torch.exp((log_psi_other-log_psi.unsqueeze(1).repeat(1, args.size)))
            Win_lambda = Win_lambda * psi_ratio
            R = R + theta 

    with torch.no_grad():
        TP_t = P_t + (torch.sum(P_t_other * Win_lambda, 1) - R * P_t) * args.delta_t
    return TP_t


def TransitionState2D(sample, args, Tstep, step, net_new, pre_trained_model,theta):
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
            print('Left-up BC.')
        Col1 = torch.cat(
            (torch.ones(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
        )  # down sites
    if args.BC == 2:
        if step == 0:
            print('All-up BC.')
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
    aa = torch.sum(Sample1D, 1)  # New code: Manual add decay operator with same order for all-0 state
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
        #if Tstep == 0 and step <= 30:
                #tilting = calculate_tilting(R,Win_lambda)
                #print(f"W_tilted: {tilting}")
    # New code 2: use logexpsum:
    with torch.no_grad():
        c = torch.as_tensor(args.c, dtype=torch.float64).to(args.device)
        if Tstep == 0:
            ones = torch.sum((Sample1D == 1.0), 1)

            Temp = torch.transpose(SampleNeighbor1DExtend, 0, 1)  
            Temp[:, 0, 0] = torch.as_tensor(1).to(args.device, dtype=default_dtype_torch)
            Temp = Temp + (Temp - 1)
            
            Temp3 = torch.reshape(Temp, (args.batch_size * args.size, args.L, args.L))  # For RNN
            # BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
            LogP_t = ones * torch.log(c) + (args.size - 1 - ones) * torch.log(1 - c)
            ones = (SampleNeighbor1D == 1.0).sum(dim=2)
            LogP_t_other = ones.t() * torch.log(c) + (args.size - 1 - ones).t() * torch.log(
                1 - c
            )  # BatchSize X NeighborSize
        else:
            LogP_t = net_new.log_prob(sample).detach()
            Temp = torch.transpose(SampleNeighbor1DExtend, 0, 1)  # fixing the first spin up
            # Set the first spin up to avoid numerical problem when generating prob
            Temp[:, 0, 0] = torch.as_tensor(1).to(args.device, dtype=default_dtype_torch)

            Temp = Temp + (Temp - 1)  # Change 0 to -1 back

            Temp3 = torch.reshape(Temp, (args.batch_size * args.size, args.L, args.L))  # For RNN

            # BatchSize X NeighborSize: checked, it is consistent with for loop
            LogP_t_other = torch.reshape(net_new.log_prob(Temp3), (args.batch_size, args.size)).detach()
            LogP_t_other = LogP_t_other[:, 1:]  # fixing the first spin up

        if args.Doob == 1:
            log_psi = pre_trained_model.log_psi(sample.to(args.device)).detach() 
            log_psi_other_temp = pre_trained_model.log_psi(Temp3.to(args.device)).detach()  
            log_psi_other = torch.reshape(log_psi_other_temp, (args.batch_size, args.size)).detach()
            psi_ratio = torch.exp((log_psi_other-log_psi.unsqueeze(1).repeat(1, args.size)))
            Win_NED = Win_lambda * psi_ratio[:, 1:]
            R = R + theta 
            Temp2 = (1 + (torch.sum(torch.exp(LogP_t_other - LogP_t.repeat(args.size - 1, 1).t()) * Win_NED, 1) - R)* args.delta_t)

        else:
            Temp2 = (1 + (torch.sum(torch.exp(LogP_t_other - LogP_t.repeat(args.size - 1, 1).t()) * Win_lambda, 1) - R)* args.delta_t)
            

    if torch.min(Temp2) < 0:
            print('reduce delta t at ', step)
            Temp2[Temp2 <= 0] = 1e-300
    LogTP_t1 = torch.log(Temp2) + LogP_t
    
    TP_t1 = torch.exp(LogTP_t1)

    #P_t = torch.exp(LogP_t)
    #P_t_other = torch.exp(LogP_t_other)
    #with torch.no_grad():
        #TP_t1 = P_t + (torch.sum(P_t_other * Win_lambda, 1) - R * P_t) * args.delta_t
    return TP_t1


def calculate_tilting(R, Win_lambda):  
    Win_lambda_col_sum1 = torch.sum(Win_lambda, dim=1)  
    total_sum1 = Win_lambda_col_sum1 - R  
    tilting = torch.mean(total_sum1)  
    return tilting

def calculate_bias(R, Win_NED):  
    Win_lambda_col_sum2 = torch.sum(Win_NED, dim=1)  
    total_sum2 = Win_lambda_col_sum2 - R  
    bias = torch.mean(total_sum2)  
    return bias