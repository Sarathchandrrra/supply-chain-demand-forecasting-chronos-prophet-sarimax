# Supply Chain Demand Forecasting – Chronos-2 vs Prophet vs SARIMAX

This repository contains a Colab notebook that compares three time series forecasting models on a synthetic supply chain demand dataset:
- SARIMAX (statsmodels)
- Prophet (Meta)
- Chronos-2 (Amazon)

## Problem

Forecast daily demand for a single SKU at a warehouse and compare model performance.

The goal is to see how a classical statistical model, an interpretable business model, and a transformer-based foundation model behave on the same data and train–test split.

## Data

The dataset is generated in the notebook.

- Daily frequency
- Around 6000 days of history starting from 2010-01-01
- Columns:
  - ds: date
  - y: demand quantity
  - promo_flag: 1 on promotion days, 0 otherwise
  - holiday_flag: 1 on selected holidays (e.g. Jan 26, Aug 15), 0 otherwise

Demand is built from:
- base level
- slow upward trend
- weekend uplift (weekly seasonality)
- promotion effect
- holiday effect
- random noise

Train–test split:
- Train: first 80% of dates
- Test: last 20% of dates

## Models

1. SARIMAX (statsmodels)
   - order = (1, 1, 1)
   - seasonal_order = (1, 1, 1, 7) for weekly pattern
   - exogenous variables: promo_flag, holiday_flag

2. Prophet
   - yearly_seasonality = True
   - weekly_seasonality = True
   - daily_seasonality = False
   - added regressors: promo_flag, holiday_flag

3. Chronos-2
   - uses Chronos2Pipeline from chronos-forecasting
   - single series id ("sku_A123")
   - context: timestamp, target, promo_flag, holiday_flag in the train period
   - future covariates: timestamp, promo_flag, holiday_flag in the test period

## Results

Metrics on the test set:

| Model                 | RMSE   | MAPE (%) |
|-----------------------|--------|----------|
| SARIMAX (statsmodels) | 10.16  | 6.08     |
| Prophet               | 10.22  | 6.15     |
| Chronos-2             | 10.84  | 6.29     |

On this synthetic series:
- SARIMAX has the lowest error.
- Prophet is very close.
- Chronos-2 is slightly worse, but all three models are in the same range (around 6–6.3% MAPE).

## Interpretation

- The data was generated with linear trend, weekly seasonality, and simple promo/holiday effects.
- This matches the assumptions of SARIMAX, so it performs slightly better.
- Prophet gives similar accuracy and adds clear separation of trend and seasonality, which is useful for explanation.
- Chronos-2 is designed for more complex, multi-series problems; on this simple single series it does not have a big advantage.

## Opening the notebook

If your environment has trouble rendering the `.ipynb` file directly (e.g., GitHub viewer errors or missing Jupyter extensions),
you can either:

- Use [nbviewer](https://nbviewer.org/) with the raw notebook URL to view it in the browser.
- Run `jupyter notebook` or `jupyter lab` locally and open `Chronos_vs_Prophet_vs_Statsmodels_on_Supply_Chain_Demand.ipynb`.
- Read the auto-generated Python script version `Chronos_vs_Prophet_vs_Statsmodels_on_Supply_Chain_Demand.py`, which mirrors each
  notebook cell with `# %%` markers and trimmed outputs so it can be inspected as plain code.

## Files

- `Chronos_vs_Prophet_vs_Statsmodels_on_Supply_Chain_Demand.ipynb`: Colab notebook with data generation, train–test split, model
  training (SARIMAX, Prophet, Chronos-2), metrics, and plots.
- `Chronos_vs_Prophet_vs_Statsmodels_on_Supply_Chain_Demand.py`: Script export of the notebook so the content is readable without
  a notebook viewer.
