"""
DR-Learner with Cross-Fitting
===============================
Implements the Doubly Robust Learner (Kennedy, 2023) with K-fold cross-fitting
for constructing pseudo-labels of individual treatment effects.

Since CatBoost is unavailable, we use GradientBoostingRegressor/Classifier
from scikit-learn as the nuisance models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


def fit_nuisance_models(X_train, t_train, y_train):
    """
    Fit the three nuisance models:
    - e(x): propensity score model P(T=1|X)
    - f0(x): outcome model E[Y|X, T=0]
    - f1(x): outcome model E[Y|X, T=1]
    """
    # Propensity score model
    propensity_model = LogisticRegression(max_iter=1000, C=1.0)
    propensity_model.fit(X_train, t_train)

    # Outcome models
    idx0 = t_train == 0
    idx1 = t_train == 1

    f0_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    f1_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )

    if idx0.sum() > 0:
        f0_model.fit(X_train[idx0], y_train[idx0])
    if idx1.sum() > 0:
        f1_model.fit(X_train[idx1], y_train[idx1])

    return propensity_model, f0_model, f1_model


def compute_dr_pseudo_labels(X, t, y, propensity_model, f0_model, f1_model, clip=0.05):
    """
    Compute DR pseudo-labels using Equation 19 from the paper:

    tau_tilde_i = (t_i - e_hat(x_i)) / (e_hat(x_i) * (1 - e_hat(x_i)))
                  * (y_i - f_{t_i}(x_i)) + f1(x_i) - f0(x_i)
    """
    n = len(y)

    # Predict propensity scores
    e_hat = propensity_model.predict_proba(X)[:, 1]
    e_hat = np.clip(e_hat, clip, 1 - clip)  # Clip for stability

    # Predict outcomes
    f0_pred = f0_model.predict(X)
    f1_pred = f1_model.predict(X)

    # Compute f_{t_i}(x_i): outcome prediction for the observed treatment
    ft_pred = np.where(t == 1, f1_pred, f0_pred)

    # DR pseudo-label (Equation 19)
    weight = (t - e_hat) / (e_hat * (1 - e_hat))
    residual = y - ft_pred
    tau_tilde = weight * residual + f1_pred - f0_pred

    return tau_tilde


def cross_fit_pseudo_labels(X, t, y, K=5, seed=42):
    """
    Algorithm 1 (Step 1-6): Cross-fitting to construct pseudo-labels.

    For each fold k:
    1. Fit nuisance models on complement data
    2. Construct DR pseudo-labels for fold k samples

    Returns:
        tau_tilde: pseudo-labels for all samples (n,)
    """
    n = len(y)
    tau_tilde = np.zeros(n)

    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    for fold_idx, (complement_idx, fold_idx_arr) in enumerate(kf.split(X)):
        # Step 3-4: Fit nuisance models on complement data
        X_comp, t_comp, y_comp = X[complement_idx], t[complement_idx], y[complement_idx]
        propensity_model, f0_model, f1_model = fit_nuisance_models(X_comp, t_comp, y_comp)

        # Step 5-6: Construct pseudo-labels for fold k
        X_fold = X[fold_idx_arr]
        t_fold = t[fold_idx_arr]
        y_fold = y[fold_idx_arr]

        tau_tilde[fold_idx_arr] = compute_dr_pseudo_labels(
            X_fold, t_fold, y_fold, propensity_model, f0_model, f1_model
        )

    return tau_tilde


def evaluate_pseudo_labels(tau_tilde, tau_true):
    """Evaluate quality of pseudo-labels against ground truth."""
    mse = np.mean((tau_tilde - tau_true) ** 2)
    corr = np.corrcoef(tau_tilde, tau_true)[0, 1]
    bias = np.mean(tau_tilde - tau_true)
    return {'mse': mse, 'correlation': corr, 'bias': bias}


if __name__ == "__main__":
    from data_generation import generate_dataset1, generate_dataset2, split_data

    for gen_func, name in [(generate_dataset1, "Dataset 1"),
                           (generate_dataset2, "Dataset 2")]:
        X, t, y, tau, prop = gen_func(n=3000)
        data = split_data(X, t, y, tau, prop)

        X_train = data['train']['X']
        t_train = data['train']['t']
        y_train = data['train']['y']
        tau_train = data['train']['tau']

        tau_tilde = cross_fit_pseudo_labels(X_train, t_train, y_train, K=5)
        metrics = evaluate_pseudo_labels(tau_tilde, tau_train)

        print(f"\n=== {name} Pseudo-Label Quality ===")
        print(f"  MSE:         {metrics['mse']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Bias:        {metrics['bias']:.4f}")
