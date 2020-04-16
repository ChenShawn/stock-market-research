# Model Benchmarks

## 1. Setting

The sequential models make predictions using 14 days historical data, together with the non-sequential data `stock_basics` obtained by `tushare`.

In current version, stock data before March 1st, 2020 are regarded as training set, and the data after are validation set. This might not be a good split point since the American market has been crashed during this specific perioid of time. The fact that `val_acc > train_acc` also confirms this point.

Early stopping is applied as the model get overfitting quite easily.

## 2. Model results

The columns `2018` and `2019` represents the percentage of profit obtained via trading simulation in the corresponding year. By defaults, simulation time starts from Dec last year to Dec this year.

Model | Version | Accuracy | Precision | Recall | 2018 | 2019 | Remarks |
--- | :--: | :--: | :---: | :---: | :---: | :---: | :---: 
SimpleSequentialModel | V1 | 86.31% | 92.37% | 90.55% | 74.33% | 52.59% | val_acc > train_acc
LuongAttentionLSTM | V1 | 85.52% | 92.52% | 89.33% |  |  | 

## 3. Embedding quality

All models are shit. Their embedding don't even make a fucking sense.
