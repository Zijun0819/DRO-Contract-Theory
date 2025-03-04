import numpy as np
from bcd_.subproblem_solver import solve_xi_subproblem, penalty


def penalty_gradient(L, gamma1, theta):
    """
    Computes the gradient of penalty(L) with respect to L.
    Recall that:
       penalty(L) = gamma1 * [ L[0]/theta[0] + sum_{j=1}^{i} (L[j]-L[j-1])/theta[j] ]
    so that:
       d/dL[i] = gamma1/theta[i]
    """
    I = len(L)
    grad = np.zeros(I)
    for i in range(0, I - 1):
        grad[i] = gamma1 / theta[i]
    return grad


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


def bcd_solver(
        L_init,  # initial guess for L
        lam_init,  # initial guess for lambda
        alpha,  # vector of alpha_i (length I)
        gamma1, gamma2, gamma3,
        theta,  # vector of theta_i (length I)
        hat_xi,  # array of hat_xi_n (length N)
        xi_lower, xi_upper,
        eps,  # epsilon in - lam * eps
        max_iter=2000,
        L_step_size=1e-1,
        lam_step_size=1e-6,
        tol=1e-3,
        console_mode=False
):
    """
    Attempt to solve:
       max_{L >= 0 monotone, lam>=0} -lam * eps + sum_{n=1}^N s_n,
     where
       s_n = min_{xi in [xi_lower, xi_upper]} [ sum_i alpha_i ln(g2 xi + g3 L_i ) + lam|xi-hat_xi_n| ]
             - alpha_sum * penalty(L).
    Using a naive BCD scheme.
    """

    N = len(hat_xi)
    I = len(L_init)
    alpha_sum = 1

    L = L_init.copy()
    lam = lam_init
    s = np.zeros(N)

    L_final = L_init.copy()
    lam_final = lam_init
    xi_final = None
    it_converge = 0
    L_list = list()
    L_list.append(L)

    # Evaluate initial objective
    obj_old = -np.inf

    def full_objective(L_, lam):
        """
        Returns the overall objective value:
             - lam * epsilon + sum_{n=1}^N s_n.
        """
        xi_stars_ = np.zeros(N)
        # pen_val = penalty(L, gamma1, theta)

        for n in range(N):
            xi_n_star, val_sub = solve_xi_subproblem(L_, lam, hat_xi[n], alpha, gamma1, gamma2, gamma3,
                                                     xi_lower, xi_upper, theta)
            xi_stars_[n] = xi_n_star
            s[n] = val_sub

        term1 = -(lam * eps)
        term2 = np.sum(s) / N
        obj_utility = term1 + term2

        return obj_utility, xi_stars_

    for it in range(max_iter):
        # --- [Block 1: Update s[n]] (and store xi_n^*)
        obj_, xi_stars = full_objective(L, lam)

        # -------- Block 2: Update L via (approximate) gradient ascent --------
        grad_L = np.zeros(I)
        for n in range(N):
            for i in range(I):
                denom = gamma2 * xi_stars[n] + gamma3 * L[i]
                grad_L[i] += alpha[i] * (gamma3 / denom)
                grad_L[i] -= alpha[i] * (gamma1 / theta[i])
        # subtract from grad_L
        grad_L /= N

        # do a step
        L_new = L + L_step_size * grad_L
        # project monotonic
        L_new = project_monotonic(L_new)

        # -------- Block 3: Update lambda via subgradient ascent --------
        # The derivative (subgradient) w.r.t. lambda is: -epsilon + sum_{n=1}^N |xi_n^* - hat_xi[n]|
        lambda_subgrad = np.average(np.abs(xi_stars - hat_xi)) - eps
        lam_new = max(lam + lam_step_size * lambda_subgrad, 0)

        # For convergence, we might recompute s (using updated L, lam)
        obj_new, _ = full_objective(L_new, lam_new)

        if abs(obj_new - obj_old) >= tol:
            lam = lam_new
            L = L_new
            obj_old = obj_new
            L_final = L
            lam_final = lam
            xi_final = xi_stars
            it_converge = it
        else:
            # lam_step_size *= 0.5
            break

        # Optionally, print progress
        if console_mode:
            print(f"Iter {it}: Obj = {obj_new:.4f}, lam = {lam:.4f}, L = {L}")
            L_list.append([round(item, 4) for item in L])

    return it_converge, L_final, lam_final, xi_final, L_list
