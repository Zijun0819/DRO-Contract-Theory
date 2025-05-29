import numpy as np
import random


class ContractModel:
    def __init__(self, config, args):
        np.random.seed(args.seed)
        random.seed(args.seed)
        len_contract = len(config.contract.theta_)
        dirichlet_beta = [1 for _ in range(len_contract)]
        self.L = np.zeros(len_contract)
        self.R = np.zeros(len_contract)
        self.gamma1 = config.model.gamma1
        self.gamma2 = config.model.gamma2
        self.gamma3 = config.model.gamma3
        self.sigma = config.model.sigma
        self.theta_ = np.asarray(config.contract.theta_)
        self.alpha_ = np.asarray(np.random.dirichlet(dirichlet_beta))
        self.xi = config.model.xi

        self.latency_calc()
        self.reward_calc()

    def latency_calc(self):
        self.L = (self.theta_*self.sigma)/self.gamma1 - (self.gamma2/self.gamma3)*self.xi

    def reward_calc(self):
        for i in range(len(self.R)):
            self.R[i] = (self.gamma1*self.L[0])/self.theta_[0]
            for j in range(1, i+1):
                self.R[i] += (self.L[j]-self.L[j-1])/self.theta_[j]

    def utility_asp(self):
        pi = self.theta_*self.R - self.gamma1*self.L

        return pi

    def utility_operator(self):
        pi_ = self.sigma*np.log(self.gamma2*self.xi+self.gamma3*self.L) - self.R
        pi = np.sum(self.alpha_*(self.sigma*np.log(self.gamma2*self.xi+self.gamma3*self.L) - self.R))

        return pi, pi_


class DROModel:
    def __init__(self, config):
        self.D = config.dro.diameter
        self.N = config.dro.sample_cnt
        self.tau = config.dro.tau

    def epsilon_calc(self):
        term1 = np.sqrt((2 / self.N) * np.log(1 / (1 - self.tau)))
        eps = self.D * term1
        return eps


