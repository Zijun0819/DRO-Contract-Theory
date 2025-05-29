import time

import numpy as np

from bcd_.bcd_solver import bcd_solver
from utils.initialization import DROModel, ContractModel
from utils.tools import get_eval_data, get_train_data, save_to_csv, avg_contract_performance, dro_contract_data_saving

import logging

# logging setting
logging.basicConfig(
    filename='running-time.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def identify_dro_lr(config, args, hat_xi, L_step_size, dro_model, console=False, console_file_name=None):
    dro_contract_model = ContractModel(config, args)
    len_contract = len(config.contract.theta_)
    L_init = [0] * len_contract
    dro_model = dro_model
    alpha = dro_contract_model.alpha_
    eps = dro_model.epsilon_calc()
    lam_step_size = np.float64(config.bcd.lam_step_size)
    tol = np.float64(config.bcd.tol)
    it_converge, L, lam, s, L_list, obj_list = bcd_solver(L_init=L_init, lam_init=config.model.lam_init, alpha=alpha,
                                                          gamma1=config.model.gamma1,
                                                          gamma2=config.model.gamma2, gamma3=config.model.gamma3,
                                                          theta=config.contract.theta_,
                                                          hat_xi=hat_xi, xi_lower=config.dro.xi_lower,
                                                          xi_upper=config.dro.xi_upper,
                                                          eps=eps,
                                                          max_iter=config.bcd.max_iter, L_step_size=L_step_size,
                                                          lam_step_size=lam_step_size, tol=tol,
                                                          console_mode=config.bcd.console_mode)
    # Contract bundles under DRO_Contract
    dro_contract_model.L = np.array(L)
    dro_contract_model.reward_calc()
    print(f"The BCD algorithm converged at {it_converge} rounds => lam={lam}, L={L}")
    if console:
        # L_list = [item for ind, item in enumerate(L_list) if ind % 10 == 0]
        # save_to_csv(L_list, filename=console_file_name or "results/identify_dro_L_list.csv")

        obj_list = [[item] for item in obj_list]
        save_to_csv(obj_list, filename=console_file_name or "results/identify_dro_obj_list.csv")

    return dro_contract_model


def identify_dro_optimal_lr(config, args):
    hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    dro_model = DROModel(config)

    vL_step_size = config.comparison.vL_step_size
    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()
    for L_step_size in vL_step_size:
        print(f"==================> The L learning rate is {L_step_size} <==================")
        L_step_size = np.float64(L_step_size)
        dro_contract = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model)

        contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                                  dro_contract,
                                                                                                  contract_item_L,
                                                                                                  contract_item_R,
                                                                                                  asp_utility,
                                                                                                  t_shift_utility)

    save_to_csv(np.transpose(np.array(contract_item_L)), filename="results/identify_dro_optimal_lr_L.csv")
    save_to_csv(np.transpose(np.array(contract_item_R)), filename="results/identify_dro_optimal_lr_R.csv")
    save_to_csv(np.transpose(np.array(asp_utility)), filename="results/identify_dro_optimal_lr_asp_utility.csv")
    save_to_csv(np.transpose(np.array(t_shift_utility)), filename="results/identify_dro_optimal_lr_t_utility.csv")


def identify_dro_optimal_n(config, args):
    L_step_size = np.float64(config.bcd.L_step_size)
    dro_model = DROModel(config)

    v_sample_cnt = config.comparison.v_sample_cnt
    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()
    for sample_cnt in v_sample_cnt:
        print(f"==================> The number of historical data used is {sample_cnt} <==================")
        hat_xi = get_train_data(config, args, sample_cnt=sample_cnt)

        dro_contract = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model)
        contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                                  dro_contract,
                                                                                                  contract_item_L,
                                                                                                  contract_item_R,
                                                                                                  asp_utility,
                                                                                                  t_shift_utility)

    save_to_csv(np.transpose(np.array(contract_item_L)), filename="results/identify_dro_optimal_n_L.csv")
    save_to_csv(np.transpose(np.array(contract_item_R)), filename="results/identify_dro_optimal_n_R.csv")
    save_to_csv(np.transpose(np.array(asp_utility)), filename="results/identify_dro_optimal_n_asp_utility.csv")
    save_to_csv(np.transpose(np.array(t_shift_utility)), filename="results/identify_dro_optimal_n_t_utility.csv")


def identify_dro_optimal_beta(config, args):
    L_step_size = np.float64(config.bcd.L_step_size)
    hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    dro_model = DROModel(config)

    v_beta = config.comparison.v_beta
    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()
    for beta in v_beta:
        print(f"==================> The confidence level of DRO, beta, used is {beta} <==================")
        dro_model.tau = beta

        dro_contract = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model)
        contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                                  dro_contract,
                                                                                                  contract_item_L,
                                                                                                  contract_item_R,
                                                                                                  asp_utility,
                                                                                                  t_shift_utility)

    save_to_csv(np.transpose(np.array(contract_item_L)), filename="results/identify_dro_optimal_beta_L.csv")
    save_to_csv(np.transpose(np.array(contract_item_R)), filename="results/identify_dro_optimal_beta_R.csv")
    save_to_csv(np.transpose(np.array(asp_utility)), filename="results/identify_dro_optimal_beta_asp_utility.csv")
    save_to_csv(np.transpose(np.array(t_shift_utility)), filename="results/identify_dro_optimal_beta_t_utility.csv")


def identify_dro_scalability(config, args):
    lc = len(config.contract.theta_)
    L_step_size = np.float64(config.bcd.L_step_size)
    hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    dro_model = DROModel(config)

    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()
    time_start = time.time()
    conv_pth = f"results/dro_obj_list_I{lc}_N{config.dro.sample_cnt}.csv"
    dro_contract = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model, console=True,
                                   console_file_name=conv_pth)
    time_end = time.time()
    elapsed_time = time_end - time_start
    logging.info(f"DRO elapsed time of {lc} contract and {config.dro.sample_cnt} samples is: {round(elapsed_time, 2)}")
    contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                              dro_contract,
                                                                                              contract_item_L,
                                                                                              contract_item_R,
                                                                                              asp_utility,
                                                                                              t_shift_utility)

    save_to_csv(np.transpose(np.array(contract_item_L)), filename=f"results/dro_scalability_L_I{lc}_N{config.dro.sample_cnt}.csv")
    save_to_csv(np.transpose(np.array(contract_item_R)), filename=f"results/dro_scalability_R_I{lc}_N{config.dro.sample_cnt}.csv")
    save_to_csv(np.transpose(np.array(asp_utility)), filename=f"results/dro_scalability_asp_utility_I{lc}_N{config.dro.sample_cnt}.csv")
    save_to_csv(np.transpose(np.array(t_shift_utility)), filename=f"results/dro_scalability_t_utility_I{lc}_N{config.dro.sample_cnt}.csv")


def identify_dro_alpha_sensitivity(config, args):
    '''
    :param config:
    :param args: varying the seed in args to control the varying of alpha, evaluating the sensitivity of DRO_Contract
    :return:
    '''
    L_step_size = np.float64(config.bcd.L_step_size)
    hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    dro_model = DROModel(config)

    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()
    dro_contract = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model)
    alpha_ = [round(item, 4) for item in dro_contract.alpha_.tolist()]
    logging.info(f"Sensitivity analysis, seed: {args.seed}, alpha: {alpha_}")
    contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                              dro_contract,
                                                                                              contract_item_L,
                                                                                              contract_item_R,
                                                                                              asp_utility,
                                                                                              t_shift_utility)

    save_to_csv(np.transpose(np.array(contract_item_L)), filename=f"results/dro_alpha{args.seed}_sensitivity_L.csv")
    save_to_csv(np.transpose(np.array(contract_item_R)), filename=f"results/dro_alpha{args.seed}_sensitivity_R.csv")
    save_to_csv(np.transpose(np.array(asp_utility)), filename=f"results/dro_alpha{args.seed}_sensitivity_asp_utility.csv")
    save_to_csv(np.transpose(np.array(t_shift_utility)), filename=f"results/dro_alpha{args.seed}_sensitivity_t_utility.csv")


def identify_dro_diameter_sensitivity(config, args):
    '''
    :param config:
    :param args: varying the Wasserstein radius (config.dro.diameter), evaluating the sensitivity of DRO_Contract
    :return:
    '''
    L_step_size = np.float64(config.bcd.L_step_size)
    hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    dro_model = DROModel(config)

    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()
    dro_contract = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model)
    contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                              dro_contract,
                                                                                              contract_item_L,
                                                                                              contract_item_R,
                                                                                              asp_utility,
                                                                                              t_shift_utility)

    save_to_csv(np.transpose(np.array(contract_item_L)), filename=f"results/dro_D{config.dro.diameter}_sensitivity_L.csv")
    save_to_csv(np.transpose(np.array(contract_item_R)), filename=f"results/dro_D{config.dro.diameter}_sensitivity_R.csv")
    save_to_csv(np.transpose(np.array(asp_utility)), filename=f"results/dro_D{config.dro.diameter}_sensitivity_asp_utility.csv")
    save_to_csv(np.transpose(np.array(t_shift_utility)), filename=f"results/dro_D{config.dro.diameter}_sensitivity_t_utility.csv")


def identify_dro_gammas_sensitivity(config, args):
    '''
    :param config:
    :param args: varying the coefficients (config.model.gamma1, gamma2, and gamma3), evaluating the sensitivity of DRO_Contract
    :return:
    '''
    L_step_size = np.float64(config.bcd.L_step_size)
    hat_xi = get_train_data(config, args, sample_cnt=config.dro.sample_cnt)
    dro_model = DROModel(config)

    contract_item_L = list()
    contract_item_R = list()
    asp_utility = list()
    t_shift_utility = list()
    dro_contract = identify_dro_lr(config, args, hat_xi, L_step_size, dro_model)
    contract_item_L, contract_item_R, asp_utility, t_shift_utility = dro_contract_data_saving(config, args,
                                                                                              dro_contract,
                                                                                              contract_item_L,
                                                                                              contract_item_R,
                                                                                              asp_utility,
                                                                                              t_shift_utility)

    save_to_csv(np.transpose(np.array(contract_item_L)), filename=f"results/dro_gammas_{config.model.gamma1}_{config.model.gamma2}_{config.model.gamma3}_sensitivity_L.csv")
    save_to_csv(np.transpose(np.array(contract_item_R)), filename=f"results/dro_gammas_{config.model.gamma1}_{config.model.gamma2}_{config.model.gamma3}_sensitivity_R.csv")
    save_to_csv(np.transpose(np.array(asp_utility)), filename=f"results/dro_gammas_{config.model.gamma1}_{config.model.gamma2}_{config.model.gamma3}_sensitivity_asp_utility.csv")
    save_to_csv(np.transpose(np.array(t_shift_utility)), filename=f"results/dro_gammas_{config.model.gamma1}_{config.model.gamma2}_{config.model.gamma3}_sensitivity_t_utility.csv")