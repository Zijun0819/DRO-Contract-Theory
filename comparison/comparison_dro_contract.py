import random

import numpy as np

from benchmarks.drl_ppo import drl_contract
from benchmarks.stochastic_programming import latency_calc
from comparison.eval_dro_contract import identify_dro_lr
from utils.initialization import ContractModel, DROModel
from utils.tools import get_train_data, dro_contract_data_saving, save_to_csv


def malicious_zero_data_fill(hat_xi, cnt_zero_fill):
    indices = random.sample(range(len(hat_xi)), cnt_zero_fill)
    for idx in indices:
        hat_xi[idx] = 1

    return hat_xi


def get_contracts(config, args, zero_fill=False, cnt_zero_fill=50):
    hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    if zero_fill:
        hat_xi = malicious_zero_data_fill(hat_xi, cnt_zero_fill)
    L_step_size = np.float64(config.bcd.L_step_size)
    dro_model = DROModel(config)
    contract_model = ContractModel(config, args)
    dro_contract_model = ContractModel(config, args)
    sp_contract_model = ContractModel(config, args)
    ro_contract_model = ContractModel(config, args)
    drl_contract_model = ContractModel(config, args)
    ro_contract_model.xi = config.model.xi_range[0]
    ro_contract_model.latency_calc()
    ro_contract_model.reward_calc()

    '''This is converged results of DRO_Contract, circumvent the need of running DRO iteration for each time,
        for the sake of time saving'''
    dro_contract_model = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model)
    # dro_contract_model.L = np.array([42.96, 72.37, 107.08, 125.07, 151.79, 166.74, 176.66, 178.63])
    # dro_contract_model.reward_calc()
    # '''====================The code of running DRL_Contract===================='''
    drl_L, drl_R = drl_contract(config, args, hat_xi, learn_steps=int(2e6), load=True)
    drl_contract_model.L = drl_L
    drl_contract_model.reward_calc()
    # drl_contract_model.L = np.array([1.0, 2.0, 2.11, 2.17, 132.02, 155.71, 156.54, 182.78])
    # drl_contract_model.reward_calc()

    L_init = [0] * config.contract.len_contract
    sp_contract_model.L = latency_calc(L_init, config, args, hat_xi)
    sp_contract_model.reward_calc()

    contract_model_list = [contract_model, dro_contract_model, sp_contract_model, ro_contract_model, drl_contract_model]

    return contract_model_list


def run_comparison(config, args, zero_fill=False, cnt_zero_fill=50):
    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()

    contract_models = get_contracts(config, args, zero_fill=zero_fill, cnt_zero_fill=cnt_zero_fill)

    for contract_model in contract_models:
        contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                                  contract_model,
                                                                                                  contract_item_L,
                                                                                                  contract_item_R,
                                                                                                  asp_utility,
                                                                                                  t_shift_utility)

    if zero_fill:
        save_to_csv(np.transpose(np.array(contract_item_L)), filename=f"results/run_comparison_L_{cnt_zero_fill}.csv")
        save_to_csv(np.transpose(np.array(contract_item_R)), filename=f"results/run_comparison_R_{cnt_zero_fill}.csv")
        save_to_csv(np.transpose(np.array(asp_utility)), filename=f"results/run_comparison_asp_utility_{cnt_zero_fill}.csv")
        save_to_csv(np.transpose(np.array(t_shift_utility)), filename=f"results/run_comparison_utility_{cnt_zero_fill}.csv")
    else:
        save_to_csv(np.transpose(np.array(contract_item_L)), filename="results/run_comparison_L.csv")
        save_to_csv(np.transpose(np.array(contract_item_R)), filename="results/run_comparison_R.csv")
        save_to_csv(np.transpose(np.array(asp_utility)), filename="results/run_comparison_asp_utility.csv")
        save_to_csv(np.transpose(np.array(t_shift_utility)), filename="results/run_comparison_utility.csv")
