import numpy as np

from bcd_.bcd_solver import bcd_solver
from utils.initialization import DROModel, ContractModel
from utils.tools import get_eval_data, get_train_data, save_to_csv, avg_contract_performance, dro_contract_data_saving


def identify_dro_lr(config, args, hat_xi, L_step_size, dro_model, console=False):
    dro_contract_model = ContractModel(config, args)
    L_init = [0] * config.contract.len_contract
    dro_model = dro_model
    alpha = dro_contract_model.alpha_
    eps = dro_model.epsilon_calc()
    lam_step_size = np.float64(config.bcd.lam_step_size)
    tol = np.float64(config.bcd.tol)
    it_converge, L, lam, s, L_list = bcd_solver(L_init=L_init, lam_init=config.model.lam_init, alpha=alpha,
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
        L_list = [item for ind, item in enumerate(L_list) if ind % 10 == 0]
        save_to_csv(L_list, filename="results/identify_dro_lr_L_list_100.csv")

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
