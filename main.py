import argparse
import csv

from comparison.eval_dro_contract import identify_dro_optimal_lr, identify_dro_lr, identify_dro_optimal_n, \
    identify_dro_optimal_beta
from comparison.comparison_dro_contract import run_comparison, malicious_zero_data_fill
from utils.tools import read_configs, obtain_sample_points, data_score_trans, \
    print_contract_info, avg_contract_performance, generate_sampled_data, save_to_csv, get_train_data
from utils.initialization import DROModel, ContractModel
import numpy as np
import random
from bcd_.bcd_solver import bcd_solver
from benchmarks.stochastic_programming import latency_calc
from benchmarks.drl_ppo import drl_contract

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRO-based contract theory for Edge AIGC Service')
    parser.add_argument("--config", default='config.yml', type=str, help="Path to the config file")
    parser.add_argument('--sample_data_pth', default=f'aigcmodel\\data\\sample_#200_score.csv', type=str,
                        help='Path for the data of uncertain performance of AIGC model')
    parser.add_argument("--eval_data_pth", default='aigcmodel\\data\\eval_#50_score.csv', type=str,
                        help="Path for the data of DRO_Contract performance evaluation")
    parser.add_argument('--seed', default=99, type=int, metavar='N', help='Random seed for initializing')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    config = read_configs(args.config)

    ''' Identify the optimal learning rate of L in the range of [1e3, 5e3, 1e4, 5e4, 1e5] '''
    # identify_dro_optimal_lr(config, args)
    ''' 
        Assess the convergence curve of the contract item L since R can be deduced via L;
        Under the setting of L_step_size = 1e4, sample_cnt = 200, beta=0.99 
    '''
    # L_step_size = np.float64(config.bcd.L_step_size)
    # hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    # # Obtain the converge curve of the contract item L when the historical data are shift
    # # hat_xi = malicious_zero_data_fill(hat_xi, cnt_zero_fill=100)
    # dro_model = DROModel(config)
    # identify_dro_lr(config, args, hat_xi=hat_xi,  L_step_size=L_step_size, dro_model=dro_model, console=True)
    ''' Identify the optimal sample count of N in the range of [10, 50, 100, 200] '''
    # identify_dro_optimal_n(config, args)
    ''' Identify the optimal DRO confidence level beta in the range of [0.8, 0.85, 0.9, 0.95, 0.99] '''
    # identify_dro_optimal_beta(config, args)

    ''' Run the comparison code with benchmarks traditional_contract, sp_contract, ro_contract, drl_contract '''
    run_comparison(config, args, zero_fill=True, cnt_zero_fill=150)
