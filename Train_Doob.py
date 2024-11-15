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
from Transition import (
    TransitionState1D,
    TransitionState2D,
)
from gru import GRU
from gru2DFirstUp import GRU2D
from args import args
from SCGF import load_pre_trained_VAN_model_1D, load_pre_trained_VAN_model_2D,local_SCGF_1D,local_SCGF_2D
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def Test():
    # #Initialize parameters: see args.py for a the help information on the parameters if needed
    # #System parameters:
    args.Model = '2DSouthEast'  # type of models: #'1DFA' or '1DEast' or '2DFA' or '2DSouthEast'
    args.L = 4  # Lattice size
    args.Tstep = 21  # Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
    args.delta_t = 0.1  # Time step length
    args.dlambda =  0.1  # The steplength from the left to the rigth boundary value of the count field  (lambda=s)
    args.dlambdaL = 0    # left boundary value of the count field  (lambda=s)
    args.dlambdaR = 0.1  # Rigth boundary value of the count field  (lambda=s)
    args.c = 0.5    #Flip-up probability
    # #Neural-network hyperparameters:
    args.net = 'rnn'  # 'rnn'#'lstm'#'rnn'Type of neural network in the VAN
    args.max_stepAll = 1500  # 0 #The epoch for the 1st  time steps
    args.max_stepLater = 100  # 00 #The epoch at time steps except for the 1st step
    args.print_step = 1  # 0   # The time step size of print and save results
    args.net_depth = 3  # 3#3  # Depth of the neural network
    args.net_width = 32  # 128 # Width of the neural network
    args.batch_size = 1000  # 1000 #batch size
    args.lr = 10e-4
    args.cuda = 0
    args.device = 'cuda'
    args.Doob = 1
    args.Hermitian = True
    # #Default parameters
    args.max_step = args.max_stepAll  # args.max_step=max_step
    args.clip_grad = 1  # clip gradient
    # args.Hermitian=False # Do Hermitian transform or not
    # args.lr_schedule=1#1, 2
    args.bias = True  # With bias or not in the neural network
    args.epsilon = 0  # 1e-6/(2**args.size) # avoid 0 value in log below
    args.clip_grad=1
    args.free_energy_batch_size = args.batch_size
    lambda_tilt_Range = 10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda))   
    start_time2 = time.time()

    if args.Model == '1DFA' or args.Model =='1DEast':
        args.size = args.L
    else:
        args.size = args.L * args.L  # the number of spin: 2D, doesnt' count the boundary spins   

    net_new = []
    
    # train the VAN
    count = -1
    for i in range(0, len(lambda_tilt_Range)):
        count += 1
        args.lambda_tilt = lambda_tilt= -lambda_tilt_Range[i]
        print('args.s:',lambda_tilt)
        theta_mean = [] 
        if args.Doob ==1 :  #load the pre-trained model
            if args.Model == '1DFA' or args.Model == '1DEast':
                pre_traind_van_model = load_pre_trained_VAN_model_1D(args)
            else:
                pre_traind_van_model = load_pre_trained_VAN_model_2D(args)

            for i in range(50):
                with torch.no_grad():
                    sample, x_hat = pre_traind_van_model.sample(args.batch_size)

                    if args.Model == '1DFA' or args.Model == '1DEast':
                        local_theta = local_SCGF_1D(sample, args, pre_traind_van_model)
                    else:
                        local_theta = local_SCGF_2D(sample, args, 1, pre_traind_van_model)

                local_theta_mean = torch.mean(local_theta).item()  
                theta_mean.append(local_theta_mean)   
            theta = sum(theta_mean) / len(theta_mean)  
            print(f"theta: {theta}") 
        
            
        else:
            pre_traind_van_model , theta = (0,0)
        
        
        
        args.max_step = args.max_stepAll
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

        else:
            params = {  
            'L': args.L,  
            'size':args.size,
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


        net.to(args.device)

        params_prime = list(net.parameters())
        params_prime = list(filter(lambda p: p.requires_grad, params_prime))
        my_log('{}\n'.format(net))
        params_prime = list(net.parameters())
        params_prime = list(filter(lambda p: p.requires_grad, params_prime))
        nparams = int(sum([np.prod(p.shape) for p in params_prime]))
        my_log('Total number of trainable parameters: {}'.format(nparams))
        optimizer = torch.optim.Adam(params_prime, lr=args.lr, betas=(0.9, 0.999))
        print_args()
        init_out_dir()
        Listloss_mean = []
        Listloss_std = []

        for Tstep in range(args.Tstep):
            # Start
                start_time = time.time()
                init_time = time.time() - start_time
                start_time = time.time()
                sample_time = 0
                train_time = 0
                if Tstep >= 1:
                    args.max_step = args.max_stepLater
                if args.lr_schedule:
                    if args.lr_schedule_type == 1:
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            factor=0.5,
                            patience=int(args.max_step * args.Percent),
                            verbose=True,
                            threshold=1e-4,
                            min_lr=1e-5,
                        )
                    if args.lr_schedule_type == 2:
                        scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer, lr_lambda=lambda epoch: 1 / (epoch * 10 * args.lr + 1)
                        )
                    if args.lr_schedule_type == 3:
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            optimizer, 10 ** (-2 / args.max_step)
                        )  # lr_final = 0.01 * lr_init (1e-2 -> 1e-4)
                    if args.lr_schedule_type == 4:
                        scheduler = torch.optim.lr_scheduler.CyclicLR(
                            optimizer,
                            base_lr=1e-4,
                            max_lr=1e-2,
                            step_size_up=20,
                            step_size_down=20,
                            mode='exp_range',
                            gamma=0.999,
                            scale_fn=None,
                            scale_mode='cycle',
                            cycle_momentum=False,
                            last_epoch=-1,
                        )
                for step in range(0, args.max_step + 1):
                    optimizer.zero_grad()
                    sample_start_time = time.time()
                    with torch.no_grad():
                        sample, _ = net.sample(args.batch_size)
                    sample_time += time.time() - sample_start_time
                    train_start_time = time.time()
                
                    log_prob = net.log_prob(sample)  # sample has size batchsize X 1 X systemSize
                    if args.Model == '1DFA' or args.Model == '1DEast':
                        if args.Doob == 1:
                            with torch.no_grad():
                                aa = TransitionState1D(sample, args, Tstep, net_new, pre_traind_van_model,theta).detach()                       
                                TP_t = torch.abs(aa)  # +torch.min(aa)*(1e-8)#args.epsilon # avoid 0 value in log below
                                TP_t_normalize = TP_t 
                        else:
                            with torch.no_grad():
                                aa = TransitionState1D(sample, args, Tstep, step, net_new, pre_traind_van_model,theta).detach().detach()
                                TP_t = torch.abs(aa)  # +torch.min(aa)*(1e-8)#args.epsilon # avoid 0 value in log below
                                TP_t_normalize = (TP_t / TP_t.sum() * (torch.exp(log_prob)).sum()).detach()
                        
                    else:
                        if args.Doob == 1:
                            with torch.no_grad():
                                aa = TransitionState2D(sample, args, Tstep, step, net_new, pre_traind_van_model,theta).detach()                       
                                TP_t = torch.abs(aa)  # +torch.min(aa)*(1e-8)#args.epsilon # avoid 0 value in log below
                                TP_t_normalize = TP_t 
                        else:
                            with torch.no_grad():
                                aa = TransitionState2D(sample, args, Tstep, step, net_new, pre_traind_van_model,theta).detach().detach()
                                TP_t = torch.abs(aa)  # +torch.min(aa)*(1e-8)#args.epsilon # avoid 0 value in log below
                                TP_t_normalize = (TP_t / TP_t.sum() * (torch.exp(log_prob)).sum()).detach()

                    loss = log_prob - torch.log(TP_t)
                    assert not TP_t.requires_grad
                    loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
                    loss_reinforce.backward()

                    if args.clip_grad:
                        nn.utils.clip_grad_norm_(params_prime, args.clip_grad)
                    optimizer.step()
                    if args.lr_schedule:
                        scheduler.step(loss.mean())
                    train_time += time.time() - train_start_time

                    loss_std = loss.std()  # /args.size
                    loss_mean = (
                        loss.mean()
                    )  # / args.size#(P_tnew * (P_tnew / (TP_t/torch.sum(TP_t))).log()).sum()
                    Listloss_mean.append(loss_mean.detach().cpu().numpy())
                    Listloss_std.append(loss_std.detach().cpu().numpy())

                    # print out:
                    if args.print_step and step % args.print_step == 0 and Tstep % int(args.print_step) == 0:
                        if step > 0:
                            sample_time /= args.print_step
                            train_time /= args.print_step
                        used_time = time.time() - start_time
                        my_log('init_time = {:.3f}'.format(init_time))
                        my_log('Training...')
                        my_log(
                            # ',DynPartiFuncFactorLog={:.8f}'#' F = {:.8g}, F_std = {:.8g}, S = {:.8g}, E = {:.8g}, M = {:.8g}, Q = {:.8g}, lr = {:.3g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                            'lambda={}, Time step of equation={}, Training step = {}, used_time = {:.3f}, loss_std={:.20f},loss_mean={}'.format(
                                lambda_tilt,
                                Tstep,
                                step,
                                used_time,
                                step,
                                torch.abs(loss_std),
                                torch.abs(loss_mean),  # DynPartiFuncFactorLog,
                            )
                        )
                        sample_time = 0
                        train_time = 0

                    with torch.no_grad():
                        if Tstep % 10 == 0 :
                            if args.out_filename is None:
                                raise ValueError("args.out_filename is None, please provide a valid filename.")

                            PATH = args.out_filename +str(Tstep) +str(lambda_tilt)+str(args.L)+str(args.Model)
                            state_dict = net.state_dict()  
                            torch.save(state_dict, PATH + '.pth')
                        net_new = copy.deepcopy(net)  # net
                        net_new.requires_grad = False

        end_time2 = time.time()
        print('Time ', (end_time2 - start_time2) / 60)
        print('Time ', (end_time2 - start_time2) / 3600)


if __name__ == '__main__':
    Test()