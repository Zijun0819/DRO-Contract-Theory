import random

import numpy as np
from scipy.optimize import fsolve


def latency_calc(L_initial, config, args, xi_hat):
    np.random.seed(args.seed)
    random.seed(args.seed)
    gamma1 = config.model.gamma1
    gamma2 = config.model.gamma2
    gamma3 = config.model.gamma3
    theta_ = config.contract.theta_
    len_contract = len(config.contract.theta_)
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
        Perform an isotonic regression on vector L.
        Return an array out of the same length as L, such that out[0] <= out[1] <= ... <= out[n-1],
        and minimize the deviation between it and L (for common “smoothing” objectives),
        or it can be regarded as the monotonic sequence closest to L.

        - enforce_positive=True: Optional. If True, the final result guarantees values >= 1e-12,
          avoiding negative or zero values in your BCD that could cause log(...) errors.

        Algorithm: Stack-based Pool Adjacent Violators Algorithm (PAVA).
        Time complexity O(n).
        """
        n = len(L)
        # Initial weights are set to 1: each element has its own “block.”
        w = np.ones(n, dtype=float)

        # Make a copy to avoid modifying the original L.
        vals = np.array(L, dtype=float)

        if enforce_positive:
            vals = np.maximum(vals, 1e-12)

        # Prepare a stack to store several (avg_value, total_weight) blocks.
        stack = []

        for i in range(n):
            # Treat the i-th element as a new block (vals[i], w[i]).
            cur_val = vals[i]
            cur_w = w[i]

            # stacking
            stack.append([cur_val, cur_w])

            # Check whether the two blocks at the top of the stack violate monotonicity. If so, merge them.
            # Note: As long as the average value of the top two blocks violates (front > back), reverse merge.
            while len(stack) > 1:
                # Two blocks on top of the stack
                v2, w2 = stack[-1]
                v1, w1 = stack[-2]

                if v1 > v2:
                    # Violation of monotonicity => Merge them into a larger block
                    new_w = w1 + w2
                    new_v = (v1 * w1 + v2 * w2) / new_w

                    # Pop out these two blocks, then push the merged block back in.
                    stack.pop()
                    stack.pop()
                    stack.append([new_v, new_w])
                else:
                    # Does not violate => Stack top is monotonic, merging can be stopped.
                    break

        # Flattening: Expand the blocks in the stack one by one into an array with length=n.
        out = np.zeros(n, dtype=float)
        idx = 0
        for (block_val, block_weight) in stack:
            # Here, block_weight is usually an integer (because it starts with 1, 1, ..., 1).
            # Furthermore, block_val is the average value of these merged elements.
            count = int(round(block_weight))
            # Fill count positions:
            out[idx: idx + count] = block_val
            idx += count

        # To ensure positive:
        if enforce_positive:
            out = np.maximum(out, 1e-12)

        return out

    # Apply the algorithm
    L_solution = project_monotonic(L_solution)

    return L_solution
