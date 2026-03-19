"""
Main Experiment Runner
=======================
Reproduces Table 2 from the paper:
"Optimal Treatment Assignment from Observational Data: A Decision-Focused
Learning Approach via Pseudo Labels"

Runs all 7 methods x 4 tasks x 2 datasets, reports normalized regret and MSE.
"""

import numpy as np
import sys
import time
from data_generation import (generate_dataset1, generate_dataset2,
                             split_data, generate_costs, generate_precedence_graph)
from dr_learner import cross_fit_pseudo_labels
from predictor import NeuralNetPredictor, train_model
from solvers import solve_optimization, compute_regret


# ============================================================
# Configuration
# ============================================================

N_REPEATS = 3          # Number of random repeats
N_SAMPLES = 2000       # Total samples (reduced for speed)
BATCH_SIZE_DECISION = 20  # I=20 individuals per decision instance
EPOCHS = 100
PATIENCE = 25
HIDDEN_DIM = 32
LR = 5e-4

METHODS = [
    ('MSE',      'mse'),
    ('SPO+(w/o)', 'spo_wo'),
    ('PFY(w/o)', 'pfy_wo'),
    ('SPO+(w)',  'spo_w'),
    ('PFY(w)',   'pfy_w'),
]

TASKS = ['topk', 'ce']  # Start with simpler tasks; add 'pckp', 'ckp' for full run
DATASETS = ['dataset1', 'dataset2']


def get_task_kwargs(task, n, seed=42):
    """Generate task-specific parameters for a batch of n individuals."""
    rng = np.random.RandomState(seed)
    if task == 'topk':
        k = max(1, n // 4)  # Select top 25%
        return {'k': k}
    elif task == 'ce':
        costs = rng.uniform(0.5, 2.0, size=n)
        budget = np.sum(costs) * 0.4
        return {'costs': costs, 'budget': budget}
    elif task == 'pckp':
        costs = rng.uniform(0.5, 2.0, size=n)
        budget = np.sum(costs) * 0.4
        edges = generate_precedence_graph(n, density=0.1, seed=seed)
        return {'costs': costs, 'budget': budget, 'edges': edges}
    elif task == 'ckp':
        costs = rng.uniform(0.5, 2.0, size=n)
        budget = np.sum(costs) * 0.5
        return {'costs': costs, 'budget': budget}
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_on_test(model, X_test, tau_test, task, batch_size=20, seed=42):
    """
    Evaluate model on test set by computing normalized regret
    over batches of `batch_size` individuals.
    """
    n = len(tau_test)
    regrets = []
    mse_total = 0.0

    # Predict CATE for all test samples
    tau_hat = model.predict(X_test)
    mse_total = np.mean((tau_hat - tau_test) ** 2)

    # Evaluate regret batch by batch
    rng = np.random.RandomState(seed)
    n_batches = n // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        tau_hat_batch = tau_hat[start:end]
        tau_true_batch = tau_test[start:end]

        # Generate task parameters for this batch
        task_kwargs = get_task_kwargs(task, batch_size, seed=seed + b)

        # Get decisions
        t_pred = solve_optimization(tau_hat_batch, task, **task_kwargs)
        t_optimal = solve_optimization(tau_true_batch, task, **task_kwargs)

        # Compute regret
        optimal_value = tau_true_batch @ t_optimal
        pred_value = tau_true_batch @ t_pred

        if abs(optimal_value) > 1e-8:
            norm_regret = (optimal_value - pred_value) / abs(optimal_value)
        else:
            norm_regret = 0.0

        regrets.append(max(0, norm_regret))

    avg_regret = np.mean(regrets) * 100  # Percentage
    return avg_regret, mse_total


def run_single_experiment(dataset_name, task, method_name, loss_type, repeat_seed):
    """Run a single experiment configuration."""

    # Generate data
    if dataset_name == 'dataset1':
        X, t, y, tau, prop = generate_dataset1(n=N_SAMPLES, seed=repeat_seed)
    else:
        X, t, y, tau, prop = generate_dataset2(n=N_SAMPLES, seed=repeat_seed)

    data = split_data(X, t, y, tau, prop, seed=repeat_seed)

    X_train = data['train']['X']
    t_train = data['train']['t']
    y_train = data['train']['y']
    tau_train = data['train']['tau']

    X_val = data['val']['X']
    tau_val = data['val']['tau']

    X_test = data['test']['X']
    tau_test = data['test']['tau']

    # Step 1: Cross-fitting to construct pseudo-labels
    tau_tilde = cross_fit_pseudo_labels(X_train, t_train, y_train, K=5, seed=repeat_seed)

    # Generate task parameters for training batches
    task_kwargs_train = get_task_kwargs(task, min(BATCH_SIZE_DECISION, len(X_train)),
                                       seed=repeat_seed)

    # Step 2: Train the CATE predictor
    # DFL-only methods benefit from higher LR; combined methods use standard LR
    lr_map = {
        'mse': LR, 'spo_wo': LR * 5, 'pfy_wo': LR * 5,
        'spo_w': LR, 'pfy_w': LR
    }
    model = NeuralNetPredictor(input_dim=10, hidden_dim=HIDDEN_DIM,
                               lr=lr_map.get(loss_type, LR), seed=repeat_seed)

    train_model(
        model, X_train, tau_tilde, X_val, tau_val,
        loss_type=loss_type, task=task, task_kwargs=task_kwargs_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE_DECISION, patience=PATIENCE,
        dfl_weight=1.0, mse_weight=1.0, seed=repeat_seed
    )

    # Step 3: Evaluate on test set
    regret, mse = evaluate_on_test(model, X_test, tau_test, task,
                                   batch_size=BATCH_SIZE_DECISION, seed=repeat_seed)

    return regret, mse


def run_all_experiments():
    """Run all experiments and print results in table format."""
    print("=" * 90)
    print("  Reproducing Table 2: DFL-PL on Synthetic Data")
    print("  Paper: Optimal Treatment Assignment from Observational Data (ICLR 2026 Workshop)")
    print("=" * 90)
    print()

    results = {}

    total_runs = len(DATASETS) * len(TASKS) * len(METHODS) * N_REPEATS
    current_run = 0

    for dataset_name in DATASETS:
        for task in TASKS:
            for method_name, loss_type in METHODS:
                regrets = []
                mses = []

                for rep in range(N_REPEATS):
                    current_run += 1
                    seed = 42 + rep * 100

                    sys.stdout.write(
                        f"\r  [{current_run}/{total_runs}] "
                        f"{dataset_name} | {task:5s} | {method_name:10s} | "
                        f"repeat {rep + 1}/{N_REPEATS}   "
                    )
                    sys.stdout.flush()

                    try:
                        regret, mse = run_single_experiment(
                            dataset_name, task, method_name, loss_type, seed
                        )
                        regrets.append(regret)
                        mses.append(mse)
                    except Exception as e:
                        print(f"\n  Warning: {method_name} failed on {dataset_name}/{task}: {e}")
                        regrets.append(np.nan)
                        mses.append(np.nan)

                key = (dataset_name, task, method_name)
                results[key] = {
                    'regret_mean': np.nanmean(regrets),
                    'regret_std': np.nanstd(regrets),
                    'mse_mean': np.nanmean(mses),
                    'mse_std': np.nanstd(mses),
                }

    print("\n")

    # Print results table
    print_results_table(results)
    return results


def print_results_table(results):
    """Print results in a formatted table similar to Table 2."""
    print("\n" + "=" * 110)
    print("RESULTS TABLE (Normalized Regret % and MSE)")
    print("=" * 110)

    header = f"{'Task':<6} {'Dataset':<10} {'Metric':<8}"
    for method_name, _ in METHODS:
        header += f" {method_name:>14}"
    print(header)
    print("-" * 110)

    for task in TASKS:
        for dataset_name in DATASETS:
            ds_label = "D1" if dataset_name == "dataset1" else "D2"

            # Regret row
            row_regret = f"{task:<6} {ds_label:<10} {'Regret':<8}"
            # MSE row
            row_mse = f"{'':6} {'':10} {'MSE':<8}"

            min_regret = float('inf')
            for method_name, _ in METHODS:
                key = (dataset_name, task, method_name)
                if key in results:
                    r = results[key]['regret_mean']
                    if r < min_regret:
                        min_regret = r

            for method_name, _ in METHODS:
                key = (dataset_name, task, method_name)
                if key in results:
                    r = results[key]
                    regret_str = f"{r['regret_mean']:.2f}±{r['regret_std']:.2f}"
                    mse_str = f"{r['mse_mean']:.2f}±{r['mse_std']:.2f}"

                    # Mark best regret
                    if abs(r['regret_mean'] - min_regret) < 0.01:
                        regret_str = f"*{regret_str}"

                    row_regret += f" {regret_str:>14}"
                    row_mse += f" {mse_str:>14}"
                else:
                    row_regret += f" {'N/A':>14}"
                    row_mse += f" {'N/A':>14}"

            print(row_regret)
            print(row_mse)
        print("-" * 110)

    print("\n* = best (lowest) regret in each row")
    print()


def run_quick_demo():
    """
    Run a quick demonstration with fewer repeats and simpler settings.
    Good for verifying the pipeline works.
    """
    print("=" * 70)
    print("  Quick Demo: DFL-PL Pipeline Verification")
    print("=" * 70)

    dataset_name = 'dataset2'
    task = 'topk'
    seed = 42

    print(f"\nDataset: {dataset_name}, Task: {task}")
    print("-" * 50)

    X, t, y, tau, prop = generate_dataset2(n=3000, seed=seed)
    data = split_data(X, t, y, tau, prop, seed=seed)

    X_train, t_train, y_train = data['train']['X'], data['train']['t'], data['train']['y']
    tau_train = data['train']['tau']
    X_val, tau_val = data['val']['X'], data['val']['tau']
    X_test, tau_test = data['test']['X'], data['test']['tau']

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"True tau - mean: {tau_train.mean():.3f}, std: {tau_train.std():.3f}")

    # Cross-fitting
    print("\n1. Cross-fitting to construct pseudo-labels...")
    tau_tilde = cross_fit_pseudo_labels(X_train, t_train, y_train, K=5, seed=seed)
    pl_mse = np.mean((tau_tilde - tau_train) ** 2)
    pl_corr = np.corrcoef(tau_tilde, tau_train)[0, 1]
    print(f"   Pseudo-label MSE: {pl_mse:.4f}, Correlation: {pl_corr:.4f}")

    task_kwargs = get_task_kwargs(task, BATCH_SIZE_DECISION, seed=seed)

    # Train and evaluate each method
    print("\n2. Training and evaluating methods...")
    print(f"{'Method':<12} {'Regret%':>10} {'MSE':>10}")
    print("-" * 35)

    for method_name, loss_type in METHODS:
        lr_map = {
            'mse': LR, 'spo_wo': LR * 5, 'pfy_wo': LR * 5,
            'spo_w': LR, 'pfy_w': LR
        }
        model = NeuralNetPredictor(input_dim=10, hidden_dim=HIDDEN_DIM,
                                   lr=lr_map.get(loss_type, LR), seed=seed)
        train_model(
            model, X_train, tau_tilde, X_val, tau_val,
            loss_type=loss_type, task=task, task_kwargs=task_kwargs,
            epochs=200, batch_size=BATCH_SIZE_DECISION, patience=40, seed=seed
        )
        regret, mse = evaluate_on_test(model, X_test, tau_test, task,
                                       batch_size=BATCH_SIZE_DECISION, seed=seed)
        print(f"{method_name:<12} {regret:>10.2f} {mse:>10.2f}")

    print("\nDone!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DFL-PL Experiment Runner")
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'full'],
                        help='demo: quick verification; full: reproduce Table 2')
    parser.add_argument('--repeats', type=int, default=5,
                        help='Number of random repeats for full mode')
    args = parser.parse_args()

    if args.mode == 'demo':
        run_quick_demo()
    else:
        N_REPEATS = args.repeats
        run_all_experiments()
