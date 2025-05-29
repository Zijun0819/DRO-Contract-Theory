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
        self.reward_history = []  # Store the historical reward
        self.window_size = 100  # Statistics for the last 100 rounds of data
        len_contract = len(config.contract.theta_)
        dirichlet_beta = [1 for _ in range(len_contract)]

        self.L = np.zeros(len_contract)
        self.R = np.zeros(len_contract)
        self.gamma1 = config.model.gamma1
        self.gamma2 = config.model.gamma2
        self.gamma3 = config.model.gamma3
        self.sigma = config.model.sigma
        self.xi_ = np.array(xi_hat)
        self.pre_reward = 0

        self.ncb = len_contract
        self.action_bound = 1
        self.theta_list = np.array(config.contract.theta_)
        self.alpha_list = np.array(np.random.dirichlet(dirichlet_beta))
        self.max_reward = -np.inf

        self.action_space = spaces.Box(
            low=-self.action_bound, high=self.action_bound, shape=(self.ncb,), dtype=np.float32
        )

        # Define observation space（state space: including previous and current latency term in the contract,
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
        """Calculate the number of contracts that meet IR constraints"""
        ir_record = np.where(self.asp_i_utility() >= 0, 1, 0)
        return ir_record

    def normalize_reward(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)  # Maintain a fixed length

        R_min, R_max = min(self.reward_history), max(self.reward_history)
        if R_max > R_min:
            reward_norm = (reward - R_min) / (R_max - R_min)
        else:
            reward_norm = 0  # If R_max == R_min, avoid division by zero errors.
        return np.clip(reward_norm, 0, 1)

    def env_reward_calc(self):
        ir_record = self.count_fit_ir()
        reward = np.sum(np.average(self.operator_utility(self.xi_), axis=0) * ir_record)
        return reward

    def enforce_monotonicity(self):
        # Sort L from smallest to largest, while adjusting R to maintain the correspondence.
        sort_idx = np.argsort(self.L)
        self.L = self.L[sort_idx]
        self.R = self.R[sort_idx]

    def step(self, action):
        """Do an action，return (obs, reward, done, truncated, info)"""
        action_vector = action

        # Calculate the new contract vector
        self.L = self.L * (1 + action_vector)

        # Handling contract restrictions (ensuring that prices and wages are within reasonable limits)
        self.L = np.clip(self.L, 1, 1000)
        self.reward_calc()
        self.enforce_monotonicity()

        # Calculate the new state
        next_state = self.L

        # Take the IR constraints into the reward calculation
        cur_reward = self.normalize_reward(self.env_reward_calc())
        # Reward for step
        discrepancy_reward = cur_reward - self.pre_reward
        # Combination reward
        reward = cur_reward + sigmoid_normalize_reward_range(discrepancy_reward)

        self.pre_reward = cur_reward

        # Set `done` as False (Contract optimization issues are usually ongoing tasks.)
        done = False
        truncated = False
        info = {}

        return next_state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment，return (obs, info)"""
        super().reset(seed=seed)
        self.L = np.random.uniform(1, 1000.0, self.ncb)
        self.reward_calc()
        self.enforce_monotonicity()

        self.pre_reward = self.normalize_reward(self.env_reward_calc())

        initial_state = self.L

        info = {}
        return initial_state, info

    def render(self):
        print(f"Current Contract Vector: {self.L} + {self.R}")


def drl_contract(config, args, xi_hat, learn_steps, load):
    set_seed(args.seed)
    final_L = None
    final_R = None
    max_cb_v = -np.inf
    env = MyEnv(config, args, xi_hat)
    load = load

    if load:
        model = PPO.load(f"checkpoints/ppo_contract_{len(config.contract.theta_)}", env=env)
    else:
        # Initialize the PPO model
        model = PPO("MlpPolicy", env, verbose=1)

        save_path = f"checkpoints/ppo_contract_{len(config.contract.theta_)}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Training the ppo model for 100,000 steps
        model.learn(total_timesteps=learn_steps)
        model.save(save_path)
        print(f"maximum reward is + {env.max_reward}")

    # Evaluation
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
