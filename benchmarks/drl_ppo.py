import os
import random
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from gymnasium import spaces
import torch


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU


def sigmoid_normalize_reward_range(reward):
    return 1 / (1 + np.exp(-reward*1000))


class MyEnv(gym.Env):
    def __init__(self, config, args, xi_hat):
        super(MyEnv, self).__init__()
        np.random.seed(args.seed)
        random.seed(args.seed)
        self.reward_history = []  # 用于存储历史 reward
        self.window_size = 100  # 统计最近 100 轮数据

        self.L = np.zeros(config.contract.len_contract)
        self.R = np.zeros(config.contract.len_contract)
        self.gamma1 = config.model.gamma1
        self.gamma2 = config.model.gamma2
        self.gamma3 = config.model.gamma3
        self.sigma = config.model.sigma
        self.xi_ = np.array(xi_hat)
        self.pre_reward = 0

        self.ncb = config.contract.len_contract
        self.action_bound = 1
        self.theta_list = np.array(config.contract.theta_)
        self.alpha_list = np.array(np.random.dirichlet(config.contract.dirichlet_beta))
        self.max_reward = -np.inf

        self.action_space = spaces.Box(
            low=-self.action_bound, high=self.action_bound, shape=(self.ncb,), dtype=np.float32
        )

        # 定义 observation space（state space: including previous and current latency term in the contract,
        # as well IR constraints keeping or breaking status）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.ncb,), dtype=np.float32
        )

    def reward_calc(self):
        for i in range(self.ncb):
            self.R[i] = (self.gamma1*self.L[0])/self.theta_list[0]
            for j in range(1, i+1):
                self.R[i] += (self.L[j]-self.L[j-1])/self.theta_list[j]

    def asp_i_utility(self):
        asp_list = self.theta_list * self.R - self.gamma1 * self.L
        return asp_list

    def operator_utility(self, xi_):
        pi_o = self.alpha_list * (self.sigma * np.log(self.gamma2 * xi_[:, np.newaxis] + self.gamma3 * self.L) - self.R)

        return pi_o

    def count_fit_ir(self):
        """计算符合 IR 约束的合同数"""
        ir_record = np.where(self.asp_i_utility() >= 0, 1, 0)
        return ir_record

    def normalize_reward(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)  # 维持固定长度

        R_min, R_max = min(self.reward_history), max(self.reward_history)
        if R_max > R_min:
            reward_norm = (reward - R_min) / (R_max - R_min)
        else:
            reward_norm = 0  # 如果 R_max == R_min，避免除 0 错误
        return np.clip(reward_norm, 0, 1)

    def env_reward_calc(self):
        ir_record = self.count_fit_ir()
        reward = np.sum(np.average(self.operator_utility(self.xi_), axis=0) * ir_record)
        return reward

    def enforce_monotonicity(self):
        # 按 L 从小到大排序，同时调整 R 保持对应关系
        sort_idx = np.argsort(self.L)
        self.L = self.L[sort_idx]
        self.R = self.R[sort_idx]

    def step(self, action):
        """执行一个动作，返回 (obs, reward, done, truncated, info)"""
        action_vector = action

        # 计算新的合同向量
        self.L = self.L * (1 + action_vector)

        # 处理合同限制（确保价格和工资在合理范围内）
        self.L = np.clip(self.L, 1, 1000)
        self.reward_calc()
        self.enforce_monotonicity()

        # 计算新的状态
        next_state = self.L

        # Take the IR constraints into the reward calculation
        cur_reward = self.normalize_reward(self.env_reward_calc())
        # Reward for step
        discrepancy_reward = cur_reward - self.pre_reward
        # Combination reward
        reward = cur_reward + sigmoid_normalize_reward_range(discrepancy_reward)

        self.pre_reward = cur_reward

        # `done` 设置为 False（合同优化问题通常是持续任务）
        done = False
        truncated = False
        info = {}

        return next_state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境，返回 (obs, info)"""
        super().reset(seed=seed)
        self.L = np.random.uniform(1, 1000.0, self.ncb)
        self.reward_calc()
        self.enforce_monotonicity()

        self.pre_reward = self.normalize_reward(self.env_reward_calc())

        initial_state = self.L

        info = {}
        return initial_state, info

    def render(self):
        """可选：打印当前合同情况"""
        print(f"Current Contract Vector: {self.L} + {self.R}")


def drl_contract(config, args, xi_hat, learn_steps, load):
    set_seed(args.seed)
    final_L = None
    final_R = None
    max_cb_v = -np.inf
    env = MyEnv(config, args, xi_hat)
    load = load

    if load:
        model = PPO.load("checkpoints/ppo_contract", env=env)
    else:
        # 初始化 PPO 模型
        model = PPO("MlpPolicy", env, verbose=1)

        save_path = "checkpoints/ppo_contract"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 训练 10 万步
        model.learn(total_timesteps=learn_steps)
        model.save(save_path)
        print(f"maximum reward is + {env.max_reward}")

    # 评估训练效果
    obs, _ = env.reset()
    for _ in range(int(1e4)):
        torch.manual_seed(args.seed)
        action, _states = model.predict(obs)
        obs, reward, _, _, _ = env.step(action)
        if env.env_reward_calc() > max_cb_v:
            max_cb_v = env.env_reward_calc()
            final_L = env.L.copy()
            final_R = env.R.copy()

    print(f"Maximum contract reward value is {round(max_cb_v, 2)}")
    return final_L, final_R
