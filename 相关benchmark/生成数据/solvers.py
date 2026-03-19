"""
Optimization Solvers
====================
Implements the four treatment assignment optimization problems from the paper:
1. Top-k Allocation
2. Cost-Efficient Allocation (CE) - 0-1 Knapsack
3. Precedence Constraint Knapsack Problem (PCKP)
4. Collapsing Knapsack Problem (CKP)

All solvers take predicted CATE values and return binary assignment vectors.
"""

import numpy as np
from itertools import combinations


# ============================================================
# Top-K Allocation
# ============================================================
def solve_topk(tau_hat, k):
    """
    max sum(tau_hat_i * t_i)  s.t. sum(t_i) <= k, t_i in {0,1}
    Solution: select the k items with largest tau_hat.
    """
    n = len(tau_hat)
    t = np.zeros(n, dtype=int)
    if k >= n:
        t[:] = 1
    else:
        top_indices = np.argsort(tau_hat)[-k:]
        t[top_indices] = 1
    return t


# ============================================================
# Cost-Efficient Allocation (CE) - 0-1 Knapsack via DP
# ============================================================
def solve_ce(tau_hat, costs, budget, scale=100):
    """
    max sum(tau_hat_i * t_i)  s.t. sum(c_i * t_i) <= B, t_i in {0,1}
    Solved via dynamic programming with discretized costs.
    """
    n = len(tau_hat)

    # Discretize costs for DP
    int_costs = np.maximum(1, np.round(costs * scale).astype(int))
    int_budget = int(np.round(budget * scale))

    # DP table
    dp = np.full(int_budget + 1, -np.inf)
    dp[0] = 0.0
    choice = np.zeros((n, int_budget + 1), dtype=bool)

    for i in range(n):
        c_i = int_costs[i]
        v_i = tau_hat[i]
        for w in range(int_budget, c_i - 1, -1):
            if dp[w - c_i] + v_i > dp[w]:
                dp[w] = dp[w - c_i] + v_i
                choice[i, w] = True

    # Backtrack
    t = np.zeros(n, dtype=int)
    w = np.argmax(dp)
    for i in range(n - 1, -1, -1):
        if choice[i, w]:
            t[i] = 1
            w -= int_costs[i]

    return t


# ============================================================
# PCKP - Precedence Constraint Knapsack Problem
# ============================================================
def solve_pckp(tau_hat, costs, budget, edges, scale=100):
    """
    max sum(tau_hat_i * t_i)
    s.t. sum(c_i * t_i) <= B
         t_m >= t_n for (m, n) in edges  (m must be selected if n is)
         t_i in {0,1}

    Uses greedy heuristic with precedence constraints.
    """
    n = len(tau_hat)
    # Build dependency: for each item, which items must be selected first
    predecessors = {i: set() for i in range(n)}
    for m, idx_n in edges:
        predecessors[idx_n].add(m)

    # Compute transitive closure of predecessors
    changed = True
    while changed:
        changed = False
        for i in range(n):
            new_preds = set()
            for p in predecessors[i]:
                new_preds |= predecessors[p]
            if not new_preds.issubset(predecessors[i]):
                predecessors[i] |= new_preds
                changed = True

    # Greedy: compute effective value (value of item + enabling future items)
    # Sort by value/cost ratio considering dependencies
    t = np.zeros(n, dtype=int)
    remaining_budget = budget
    selected = set()

    # Compute cost of selecting an item (including all prerequisites)
    def selection_cost(item):
        needed = predecessors[item] - selected
        total_cost = costs[item] if item not in selected else 0
        for p in needed:
            total_cost += costs[p]
        return total_cost

    def selection_value(item):
        needed = predecessors[item] - selected
        total_value = tau_hat[item]
        for p in needed:
            total_value += max(0, tau_hat[p])
        return total_value

    available = set(range(n)) - selected
    while available:
        best_item = None
        best_ratio = -np.inf

        for i in available:
            sc = selection_cost(i)
            if sc <= remaining_budget and sc > 0:
                ratio = selection_value(i) / sc
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_item = i

        if best_item is None:
            break

        # Select the item and all its prerequisites
        to_select = (predecessors[best_item] - selected) | {best_item}
        cost_needed = sum(costs[j] for j in to_select if j not in selected)

        if cost_needed <= remaining_budget:
            for j in to_select:
                if j not in selected:
                    t[j] = 1
                    selected.add(j)
                    remaining_budget -= costs[j]
            available -= to_select
        else:
            available.discard(best_item)

    return t


# ============================================================
# CKP - Collapsing Knapsack Problem
# ============================================================
def collapsing_capacity(n_selected, base_budget):
    """
    Capacity decreases as more items are selected.
    g(k) = base_budget * (1 - 0.3 * k / n_total)
    """
    return base_budget * max(0.1, 1 - 0.02 * n_selected)


def solve_ckp(tau_hat, costs, budget, scale=100):
    """
    max sum(tau_hat_i * t_i)
    s.t. sum(c_i * t_i) <= g(sum(t_i))  -- capacity collapses
         t_i in {0,1}

    Uses greedy heuristic since the collapsing constraint makes DP harder.
    """
    n = len(tau_hat)
    t = np.zeros(n, dtype=int)

    # Greedy approach: sort by value/cost ratio, add items if feasible
    ratios = tau_hat / (costs + 1e-8)
    order = np.argsort(ratios)[::-1]

    total_cost = 0.0
    n_selected = 0

    for i in order:
        if tau_hat[i] <= 0:
            continue
        new_n = n_selected + 1
        new_cost = total_cost + costs[i]
        effective_budget = collapsing_capacity(new_n, budget)
        if new_cost <= effective_budget:
            t[i] = 1
            total_cost = new_cost
            n_selected = new_n

    return t


# ============================================================
# Unified solver interface
# ============================================================
def solve_optimization(tau_hat, task, **kwargs):
    """
    Unified interface for all optimization problems.

    Args:
        tau_hat: predicted CATE values (n,)
        task: one of 'topk', 'ce', 'pckp', 'ckp'
        **kwargs: task-specific parameters
    Returns:
        t: binary assignment vector (n,)
    """
    if task == 'topk':
        return solve_topk(tau_hat, kwargs['k'])
    elif task == 'ce':
        return solve_ce(tau_hat, kwargs['costs'], kwargs['budget'])
    elif task == 'pckp':
        return solve_pckp(tau_hat, kwargs['costs'], kwargs['budget'], kwargs['edges'])
    elif task == 'ckp':
        return solve_ckp(tau_hat, kwargs['costs'], kwargs['budget'])
    else:
        raise ValueError(f"Unknown task: {task}")


def compute_regret(tau_true, t_pred, t_optimal):
    """
    Compute regret = tau^T * t*(tau) - tau^T * t*(tau_hat)
    """
    return tau_true @ t_optimal - tau_true @ t_pred


def compute_normalized_regret(tau_true, t_pred, task, **kwargs):
    """Compute normalized regret for a batch."""
    t_optimal = solve_optimization(tau_true, task, **kwargs)
    optimal_value = tau_true @ t_optimal
    pred_value = tau_true @ t_pred
    if abs(optimal_value) < 1e-8:
        return 0.0
    return (optimal_value - pred_value) / abs(optimal_value)


if __name__ == "__main__":
    np.random.seed(42)
    n = 20
    tau = np.random.randn(n) * 2 + 1
    costs = np.random.uniform(0.5, 2.0, n)
    budget = np.sum(costs) * 0.4
    edges = [(0, 1), (2, 3), (1, 4)]

    for task_name in ['topk', 'ce', 'pckp', 'ckp']:
        kw = {}
        if task_name == 'topk':
            kw = {'k': 5}
        elif task_name == 'ce':
            kw = {'costs': costs, 'budget': budget}
        elif task_name == 'pckp':
            kw = {'costs': costs, 'budget': budget, 'edges': edges}
        elif task_name == 'ckp':
            kw = {'costs': costs, 'budget': budget}

        t_opt = solve_optimization(tau, task_name, **kw)
        print(f"{task_name}: selected {t_opt.sum()} items, "
              f"total effect = {tau @ t_opt:.3f}")
