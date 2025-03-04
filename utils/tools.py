import argparse
import csv
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib
matplotlib.use('TkAgg')  # 切换后端，解决 PyCharm 兼容性问题


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def read_configs(file_name):
    with open(os.path.join("config", file_name), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return new_config


def obtain_sample_points(data_pth, sample_cnt=200, cnt_each_category=40) -> list:
    '''
    This method is used to collect the data of uncertain performance of the AIGC model, which can be applied to sample
    data and evaluation data. The sample data is used for solving DRO_Contract Theory Problem and the evaluation data is
    used for proposed method performance assessment.
    :param data_pth: path of sample data or evaluation data.
    :param sample_cnt: 200 for sample data and 50 for evaluation data.
    :param cnt_each_category: i.i.d sampling process, sample count for each category, the maximum value for this
    parameter regarding sample data and evaluation data are 40 and 10, respectively. When we set this parameter value as
    [2, 4, 6, 10, 20, 30, 40], the total number of sample data will be [10, 20, 30, 50, 100, 150, 200].
    :return: sampled data of the uncertain performance of the AIGC model.
    '''
    sample_cnt_each_category = int(sample_cnt / 5)
    data_list = []
    return_list = []
    with open(data_pth, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data_dict = dict()
            data_dict[row[0]] = float(row[1])  # 将值转换为浮点数
            data_list.append(data_dict)

    for i in range(5):
        return_list += data_list[i * cnt_each_category:i * cnt_each_category + sample_cnt_each_category]

    return return_list


def data_score_trans(data_list, minus_term=0):
    sample_data_score = [round(list(item.values())[0] * 50 - minus_term, 2) for item in data_list]

    return sample_data_score


def save_to_csv(data, filename):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)  # 按列写入


def print_contract_info(contract_model, color=1, N=200):
    # print(contract_model.alpha_)
    data = [round(item, 2) for item in contract_model.utility_asp().tolist()]
    print(f"utility of different types of asps: {data}")
    data_as_columns = [[item] for item in data]
    # save_to_csv(data_as_columns, f"results\\asp_utility_{N}.csv")

    color_map = {
        1: "\033[91m",  # Red
        2: "\033[92m",  # Green
        3: "\033[94m",  # Blue
        4: "\033[96m",  # Cyan
    }
    color_code = color_map.get(color, "\033[0m")  # Default to reset if color is not in map
    contract_L = [round(item, 2) for item in contract_model.L.tolist()]
    contract_R = [round(item, 2) for item in contract_model.R.tolist()]
    data_as_columns = [[item] for item in contract_L]
    # save_to_csv(data_as_columns, f"results\\latency_value_{N}.csv")
    print(f"{color_code}contract bundle of L and R: {contract_L} \t {contract_R}\033[0m")


def avg_contract_performance(contract_model, eval_xi):
    contract_score_list = []
    pi_list = []
    for xi in eval_xi:
        contract_model.xi = xi
        xi_utility_operator, pi_ = contract_model.utility_operator()
        contract_score_list.append(xi_utility_operator.item())
        pi_list.append(pi_)

    contract_avg_score = round(sum(contract_score_list) / len(contract_score_list), 2)

    pi_avg = np.average(np.asarray(pi_list), axis=0)

    return contract_score_list, contract_avg_score, pi_avg


def generate_sampled_data(a, b, size):
    np.random.seed(42)
    # 生成 beta 分布的随机数
    beta_samples = beta.rvs(a, b, size=size)

    # 将 beta 分布样本映射到 [60, 100] 区间
    low, high = 60, 100
    scaled_samples = low + (high - low) * beta_samples

    # 绘制采样数据的分布图（直方图）
    # plt.figure(figsize=(8, 5))
    # plt.hist(scaled_samples, bins=100, density=True, alpha=0.6, color='blue', edgecolor='black', label="Sampled Data")
    #
    # plt.xlabel("Value")
    # plt.ylabel("Density")
    # plt.title("Histogram of Sampled Data and Beta PDF")
    # plt.legend()
    # plt.grid()
    #
    # # 显示图像
    # plt.show()
    sampled_data = scaled_samples.tolist()
    sampled_data = [round(item, 2) for item in sampled_data]

    return sampled_data


def get_eval_data(args, eval_shift):
    eval_data_list = obtain_sample_points(data_pth=args.eval_data_pth, sample_cnt=50, cnt_each_category=10)
    eval_xi = data_score_trans(eval_data_list, minus_term=eval_shift)

    return eval_xi


def get_train_data(config, args, sample_cnt):
    sample_data_list = obtain_sample_points(data_pth=args.sample_data_pth, sample_cnt=sample_cnt,
                                            cnt_each_category=int(sample_cnt / 5))

    hat_xi = data_score_trans(sample_data_list)

    return hat_xi


def dro_contract_data_saving(config, args, dro_contract, contract_item_L, contract_item_R, asp_utility, t_shift_utility):
    shift_extent = config.comparison.shift_extent

    contract_item_L.append([round(item, 2) for item in dro_contract.L.tolist()])
    contract_item_R.append([round(item, 2) for item in dro_contract.R.tolist()])
    asp_utility.append([round(item, 2) for item in dro_contract.utility_asp().tolist()])

    t_utility_list = list()
    for shift_value in shift_extent:
        eval_xi = get_eval_data(args, shift_value)
        _, t_utility, _ = avg_contract_performance(dro_contract, eval_xi)
        t_utility_list.append(t_utility)

    t_shift_utility.append([round(item, 2) for item in t_utility_list])

    return contract_item_L, contract_item_R, asp_utility, t_shift_utility
