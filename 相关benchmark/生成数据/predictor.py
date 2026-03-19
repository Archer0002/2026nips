"""
Neural Network CATE Predictor
===============================
A simple 3-layer MLP for CATE prediction, implemented in pure numpy.
Supports training with:
- MSE loss only (standard DR-learner)
- SPO+ loss only
- PFY loss only
- SPO+ + MSE weighted combination
- PFY + MSE weighted combination

Architecture: Input(10) -> Dense(32, ReLU) -> Dense(32, ReLU) -> Dense(1)

Key insight from the paper: DFL requires treating each mini-batch as a
decision instance and solving the optimization problem within each batch.
"""

import numpy as np
from solvers import solve_optimization


class NeuralNetPredictor:
    """3-layer neural network for CATE prediction."""

    def __init__(self, input_dim=10, hidden_dim=32, lr=5e-4, seed=42):
        self.lr = lr
        self.rng = np.random.RandomState(seed)
        self.W1 = self.rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = self.rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(1)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3
        output = z3.flatten()
        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return output, cache

    def predict(self, X):
        output, _ = self.forward(X)
        return output

    def backward(self, d_output, cache):
        X = cache['X']
        z1, a1, z2, a2 = cache['z1'], cache['a1'], cache['z2'], cache['a2']
        n = X.shape[0]

        d_z3 = d_output.reshape(-1, 1)
        d_W3 = a2.T @ d_z3 / n
        d_b3 = d_z3.mean(axis=0)
        d_a2 = d_z3 @ self.W3.T

        d_z2 = d_a2 * (z2 > 0).astype(float)
        d_W2 = a1.T @ d_z2 / n
        d_b2 = d_z2.mean(axis=0)
        d_a1 = d_z2 @ self.W2.T

        d_z1 = d_a1 * (z1 > 0).astype(float)
        d_W1 = X.T @ d_z1 / n
        d_b1 = d_z1.mean(axis=0)

        max_norm = 5.0
        for g in [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]:
            norm = np.linalg.norm(g)
            if norm > max_norm:
                g *= max_norm / norm

        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3

    def copy_params(self):
        return {k: getattr(self, k).copy() for k in ['W1','b1','W2','b2','W3','b3']}

    def load_params(self, params):
        for k, v in params.items():
            setattr(self, k, v.copy())


# ============================================================
# Loss Functions
# ============================================================

def mse_loss_grad(tau_hat, tau_tilde):
    """MSE loss and gradient."""
    residual = tau_hat - tau_tilde
    loss = np.mean(residual ** 2)
    grad = 2 * residual / len(residual)
    return loss, grad


def spo_plus_loss_grad(tau_hat, tau_tilde, task, task_kwargs):
    """
    SPO+ loss for maximization (Equation 21, adapted).

    For max problem: the SPO+ surrogate upper bound on regret is:
    L = max_{t in S} (2*tau_hat - tau_tilde)^T t
        - 2*tau_hat^T t*(tau_tilde) + tau_tilde^T t*(tau_tilde)

    Subgradient w.r.t. tau_hat:
        2*(t*(2*tau_hat - tau_tilde) - t*(tau_tilde))
    """
    n = len(tau_hat)
    t_star_tilde = solve_optimization(tau_tilde, task, **task_kwargs)

    modified = 2 * tau_hat - tau_tilde
    t_star_mod = solve_optimization(modified, task, **task_kwargs)

    loss = (modified @ t_star_mod
            - 2 * tau_hat @ t_star_tilde
            + tau_tilde @ t_star_tilde)
    loss = max(0.0, loss)

    # Subgradient w.r.t. tau_hat (for gradient descent, we want to minimize)
    grad = 2.0 * (t_star_mod - t_star_tilde) / n
    return loss, grad


def pfy_loss_grad(tau_hat, tau_tilde, task, task_kwargs, sigma=0.5, n_samples=5, rng=None):
    """
    Perturbed Fenchel-Young loss (Equations 23-25).
    Gradient: E[t*(tau_hat + sigma*eps)] - t*(tau_tilde)  (normalized)
    """
    if rng is None:
        rng = np.random.RandomState()

    n = len(tau_hat)
    t_star_tilde = solve_optimization(tau_tilde, task, **task_kwargs)

    t_mean = np.zeros(n)
    for _ in range(n_samples):
        eps = rng.randn(n)
        perturbed = tau_hat + sigma * eps
        t_sample = solve_optimization(perturbed, task, **task_kwargs)
        t_mean += t_sample
    t_mean /= n_samples

    grad = (t_mean - t_star_tilde) / n
    loss = max(0.0, tau_tilde @ t_star_tilde - tau_tilde @ t_mean)
    return loss, grad


# ============================================================
# Training
# ============================================================

def get_batch_task_kwargs(task, batch_size, seed):
    """Generate task-specific parameters for a batch."""
    from data_generation import generate_precedence_graph
    rng = np.random.RandomState(seed)

    if task == 'topk':
        return {'k': max(1, batch_size // 4)}
    elif task == 'ce':
        costs = rng.uniform(0.5, 2.0, size=batch_size)
        budget = np.sum(costs) * 0.4
        return {'costs': costs, 'budget': budget}
    elif task == 'pckp':
        costs = rng.uniform(0.5, 2.0, size=batch_size)
        budget = np.sum(costs) * 0.4
        edges = generate_precedence_graph(batch_size, density=0.1, seed=seed)
        return {'costs': costs, 'budget': budget, 'edges': edges}
    elif task == 'ckp':
        costs = rng.uniform(0.5, 2.0, size=batch_size)
        budget = np.sum(costs) * 0.5
        return {'costs': costs, 'budget': budget}
    return {}


def train_model(model, X_train, tau_tilde, X_val, tau_val_true,
                loss_type='mse', task='topk', task_kwargs=None,
                epochs=500, batch_size=20, patience=50,
                dfl_weight=1.0, mse_weight=1.0, seed=42):
    """
    Train the neural network model.

    For DFL methods, each mini-batch of size `batch_size` (I=20) is treated as
    a decision instance. The optimization problem is solved within each batch.
    """
    if task_kwargs is None:
        task_kwargs = {}

    rng = np.random.RandomState(seed)
    n = len(tau_tilde)
    best_val_loss = np.inf
    best_params = model.copy_params()
    no_improve = 0
    history = {'train_loss': [], 'val_mse': []}

    # For MSE, larger batches are fine; for DFL, use decision batch size
    if loss_type == 'mse':
        effective_batch = 64
    else:
        effective_batch = batch_size  # I=20 as in paper

    # Data augmentation: shuffle-based (as mentioned in the paper, Section C.0.3)
    # Create augmented training data by multiple random shuffles
    n_augment = 2 if loss_type != 'mse' else 1
    X_aug = np.tile(X_train, (n_augment, 1))
    tau_aug = np.tile(tau_tilde, n_augment)
    n_aug = len(tau_aug)

    for epoch in range(epochs):
        perm = rng.permutation(n_aug)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_aug - effective_batch + 1, effective_batch):
            idx = perm[start:start + effective_batch]
            X_batch = X_aug[idx]
            tau_tilde_batch = tau_aug[idx]

            tau_hat, cache = model.forward(X_batch)

            # Task params for this specific batch (different each batch for diversity)
            bkw = get_batch_task_kwargs(task, len(idx), seed=seed + epoch * 1000 + start)

            if loss_type == 'mse':
                loss, grad = mse_loss_grad(tau_hat, tau_tilde_batch)

            elif loss_type == 'spo_wo':
                loss, grad = spo_plus_loss_grad(tau_hat, tau_tilde_batch, task, bkw)

            elif loss_type == 'pfy_wo':
                loss, grad = pfy_loss_grad(tau_hat, tau_tilde_batch, task, bkw, rng=rng)

            elif loss_type == 'spo_w':
                l_d, g_d = spo_plus_loss_grad(tau_hat, tau_tilde_batch, task, bkw)
                l_m, g_m = mse_loss_grad(tau_hat, tau_tilde_batch)
                loss = dfl_weight * l_d + mse_weight * l_m
                grad = dfl_weight * g_d + mse_weight * g_m

            elif loss_type == 'pfy_w':
                l_d, g_d = pfy_loss_grad(tau_hat, tau_tilde_batch, task, bkw, rng=rng)
                l_m, g_m = mse_loss_grad(tau_hat, tau_tilde_batch)
                loss = dfl_weight * l_d + mse_weight * l_m
                grad = dfl_weight * g_d + mse_weight * g_m

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            model.backward(grad, cache)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        val_pred = model.predict(X_val)
        val_mse = np.mean((val_pred - tau_val_true) ** 2)
        history['val_mse'].append(val_mse)

        # Early stopping
        monitor = val_mse if loss_type == 'mse' else avg_loss
        if monitor < best_val_loss - 1e-6:
            best_val_loss = monitor
            best_params = model.copy_params()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_params(best_params)
    return history
