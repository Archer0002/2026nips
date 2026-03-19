# PUC Released Code

This repository contains the official released code for Rethinking Causal Ranking: A Balanced Perspective on Uplift Model Evaluation.


## 📌 Overview

This project includes:

- **PUC metric** for evaluating uplift models.
- **PTONet** including hyperparameter tuning and final experiments.


## 🧪 Compare PUC with Other Metrics

To test and compare **PUC** with other metrics:

```
Open and run `metric_test.ipynb`
```



This notebook includes direct evaluation and comparison between PUC and other existing metrics.


## ⚙️ Train and Evaluate PTONet

To train **PTONet** and evaluate its performance using PUC and other metrics, follow the steps below:

### 1. Configure Hyperparameter Ranges

Open the following file:

```
tune_mulp.py
```

Inside, **set the hyperparameter ranges** you want to tune.

### 2. Run Hyperparameter Tuning

```
python tune_mulp.py
```

This script will:

- Automatically and **in parallel** run `main_synthetic_ptonet.py` for each hyperparameter combination.

- Save results to the **log folder**.

### 3. Select Best Parameters

From the generated logs, choose the best-performing parameter configuration.

Then, input those parameters into:

```
main_synthetic_ptonet.py
```

### 4. Run Final Experiments

Execute:

```
python run_exps_mulp.py
```

This script will:

- Run PTONet **50 times using different random seeds**.

- Record all training results.

### 5. Read and Summarize Results

Open `read_results.py`, modify the log path:

```
path = ''
```

Then run:

```
python read_results.py
```

You will obtain the final performance of PTONet.

## 📣 Citation

Coming soon...
