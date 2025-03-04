import random

import numpy as np
from scipy.optimize import fsolve


def latency_calc(L_initial, config, args, xi_hat):
    np.random.seed(args.seed)
    random.seed(args.seed)
    gamma1 = config.model.gamma1
    gamma2 = config.model.gamma2
    gamma3 = config.model.gamma3
    alpha_ = np.random.dirichlet(config.contract.dirichlet_beta)
    theta_ = config.contract.theta_
    len_contract = config.contract.len_contract
    N = config.dro.sample_cnt
    # N = config.data.size
    xi_hat = np.asarray(xi_hat)

    L_solution = list()

    for i in range(len_contract):
        # Define the function to solve F(L1) = 0
        def equation(Li):
            sum_term = np.sum(1 / (gamma2 * xi_hat + gamma3 * Li))
            return sum_term - (N * gamma1) / (theta_[i] * gamma3)

        Li_solution = fsolve(equation, L_initial[i])[0]

        L_solution.append(Li_solution)

        # Apply Bunching and Ironing to enforce monotonicity

    def project_monotonic(L, enforce_positive=True):
        """
        对向量 L 做单调非递减投影 (isotonic regression).
        返回一个与 L 等长的数组 out, 满足 out[0] <= out[1] <= ... <= out[n-1],
        并且使得它与 L 的偏差最小(对于常见的“平滑”目标)，
        或可视为对 L 最贴近的单调序列。

        - enforce_positive=True: 可选, 若为 True 则最终结果会保证数值 >= 1e-12，
          避免在你的 BCD 中出现负值或 0 值导致 log(...) 出错.

        算法：栈式 Pool Adjacent Violators Algorithm (PAVA).
        时间复杂度 O(n).
        """
        n = len(L)
        # 初始权重都设为 1：表示每个元素各自一个“块”
        w = np.ones(n, dtype=float)

        # 复制一份, 避免修改原 L
        vals = np.array(L, dtype=float)

        # 如果要严格 positivity, 可以先 clip:
        if enforce_positive:
            vals = np.maximum(vals, 1e-12)

        # 准备一个 stack，存放若干个 (avg_value, total_weight) 块
        stack = []

        for i in range(n):
            # 把第 i 个元素当作一个新的小块 (vals[i], w[i])
            cur_val = vals[i]
            cur_w = w[i]

            # 入栈
            stack.append([cur_val, cur_w])

            # 检查栈顶两个块是否违背单调性，若是则合并
            # 注意: 只要栈顶两个块的平均值违背 (前 > 后), 就反复合并
            while len(stack) > 1:
                # 栈顶两个块
                v2, w2 = stack[-1]  # 后面的块
                v1, w1 = stack[-2]  # 前面的块

                if v1 > v2:
                    # 违反单调 => 合并它们成一个更大的块
                    new_w = w1 + w2
                    new_v = (v1 * w1 + v2 * w2) / new_w

                    # 弹出这两个块, 再把合并块压回去
                    stack.pop()
                    stack.pop()
                    stack.append([new_v, new_w])
                else:
                    # 不违反 => 栈顶已单调, 可以停止合并
                    break

        # 扁平化：把栈里的块依次展开回 length=n 的数组
        out = np.zeros(n, dtype=float)
        idx = 0
        for (block_val, block_weight) in stack:
            # 这里 block_weight 通常是整数（因为最开始是 1, 1, ..., 1）
            # 并且 block_val 是这些被合并元素的平均值
            count = int(round(block_weight))
            # 填充 count 个位置:
            out[idx: idx + count] = block_val
            idx += count

        # 如果要再保证正数:
        if enforce_positive:
            out = np.maximum(out, 1e-12)

        return out

    # Apply the algorithm
    L_solution = project_monotonic(L_solution)

    return L_solution
