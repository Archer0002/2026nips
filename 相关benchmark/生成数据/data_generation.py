"""
Data Generation Module
=======================
Implements Dataset 1 and Dataset 2 from the paper:
"Optimal Treatment Assignment from Observational Data: A Decision-Focused
Learning Approach via Pseudo Labels" (ICLR 2026 Workshop)

Dataset 1: Large tau variation, simpler baseline
Dataset 2: Complex nonlinear relationships (Athey & Wager style)
"""

import numpy as np


def generate_dataset1(n=3000, seed=42):
    """
    Dataset 1 (Kamran et al., 2024 style):
        x_i ~ N(0, I_{10x10})
        t_i | x_i ~ Bern(sigmoid(x_{i,3}))
        eps_i ~ N(0, 1)
        tau_i = (max(x1,0) + max(x2,0) + x4^2 + |x6|^{3/2}) (large variation)
        y_i = t_i * tau_i + eps_i + max(0, x3+x4) + |x5| + x6*x7
    """
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(np.zeros(10), np.eye(10), size=n)

    # Propensity: sigmoid(x3), note 0-indexed: x_{i,3} -> X[:, 2]
    propensity = 1.0 / (1.0 + np.exp(-X[:, 2]))
    t = rng.binomial(1, propensity)

    eps = rng.normal(0, 1, size=n)

    # CATE: tau (0-indexed: x1->0, x2->1, x4->3, x6->5)
    tau = (np.maximum(X[:, 0], 0) + np.maximum(X[:, 1], 0)
           + X[:, 3] ** 2 + np.abs(X[:, 5]) ** 1.5)

    # Baseline (0-indexed: x3->2, x4->3, x5->4, x6->5, x7->6)
    baseline = (np.maximum(0, X[:, 2] + X[:, 3])
                + np.abs(X[:, 4]) + X[:, 5] * X[:, 6])

    y = t * tau + eps + baseline

    return X, t, y, tau, propensity


def generate_dataset2(n=3000, seed=42):
    """
    Dataset 2 (Athey & Wager, 2021 style):
        x_i ~ N(0, I_{10x10})
        t_i | x_i ~ Bern(sigmoid(x_{i,3}))
        eps_i ~ N(0, 1)
        tau_i = 1 + 2|x_{i,4}| + x_{i,10}^2
        y_i = t_i * tau_i + eps_i + 5*(2 + 0.5*sin(pi*x1) - 0.5*x2 + 0.75*x3*x9)
    """
    rng = np.random.RandomState(seed)
    X = rng.multivariate_normal(np.zeros(10), np.eye(10), size=n)

    # Propensity: sigmoid(x3), 0-indexed: x_{i,3} -> X[:, 2]
    propensity = 1.0 / (1.0 + np.exp(-X[:, 2]))
    t = rng.binomial(1, propensity)

    eps = rng.normal(0, 1, size=n)

    # CATE: tau (0-indexed: x4->3, x10->9)
    tau = 1 + 2 * np.abs(X[:, 3]) + X[:, 9] ** 2

    # Baseline (0-indexed: x1->0, x2->1, x3->2, x9->8)
    baseline = 5 * (2 + 0.5 * np.sin(np.pi * X[:, 0])
                    - 0.5 * X[:, 1] + 0.75 * X[:, 2] * X[:, 8])

    y = t * tau + eps + baseline

    return X, t, y, tau, propensity


def split_data(X, t, y, tau, propensity, train_ratio=2, val_ratio=1, test_ratio=3, seed=42):
    """
    Split data into train/val/test sets with ratio 2:1:3 as in the paper.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    total_ratio = train_ratio + val_ratio + test_ratio

    indices = rng.permutation(n)
    n_train = int(n * train_ratio / total_ratio)
    n_val = int(n * val_ratio / total_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    data = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        data[name] = {
            'X': X[idx], 't': t[idx], 'y': y[idx],
            'tau': tau[idx], 'propensity': propensity[idx]
        }
    return data


def generate_costs(n, seed=42):
    """Generate individual-specific treatment costs for CE/PCKP/CKP problems."""
    rng = np.random.RandomState(seed)
    costs = rng.uniform(0.5, 2.0, size=n)
    return costs


def generate_precedence_graph(n, density=0.15, seed=42):
    """
    Generate a DAG for PCKP problem.
    Returns list of (m, n) pairs meaning item m must be selected before item n.
    """
    rng = np.random.RandomState(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                edges.append((i, j))
    return edges


if __name__ == "__main__":
    for gen_func, name in [(generate_dataset1, "Dataset 1"), (generate_dataset2, "Dataset 2")]:
        X, t, y, tau, prop = gen_func(n=3000)
        data = split_data(X, t, y, tau, prop)
        print(f"\n=== {name} ===")
        print(f"Total samples: {len(y)}")
        print(f"  Train: {len(data['train']['y'])}, Val: {len(data['val']['y'])}, "
              f"Test: {len(data['test']['y'])}")
        print(f"  Treatment rate: {t.mean():.3f}")
        print(f"  tau  - mean: {tau.mean():.3f}, std: {tau.std():.3f}")
        print(f"  y    - mean: {y.mean():.3f}, std: {y.std():.3f}")
