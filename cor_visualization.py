import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import coint

tickers = [
    "KO",
    "PEP",
    "SPY",
    "QQQ",
    "DJI",
    "TSLA",
    "MSFT",
    "AAPL",
    "GOOG",
    "JPM",
    "GS"
]

start_date = "2014-01-01"
end_date = "2015-01-01"
df = yf.download(
    tickers, 
    start=start_date, 
    end=end_date, 
    auto_adjust=False
).Close
df = df.dropna()

n = len(tickers)
score_matrix = np.zeros((n, n))
pvalue_matrix = np.ones((n, n))
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        score, pval, _ = coint(df.iloc[:, i], df.iloc[:, j])
        score_matrix[i, j] = score
        pvalue_matrix[i, j] = pval
        if pval < 0.10:
            pairs.append((tickers[i], tickers[j]))

mask = np.triu(np.ones_like(pvalue_matrix, dtype=bool), k=1)
upper_vals = pvalue_matrix[mask]
min_idx_flat = np.argmin(upper_vals)
min_p = upper_vals[min_idx_flat]
idx_pairs = np.column_stack(np.where(mask))
i, j = idx_pairs[min_idx_flat]

S1, S2 = df.iloc[:, i], df.iloc[:, j]

score, pvalue, _ = coint(S1, S2)
print(f"tickers with lowest p-value: {tickers[i]} x {tickers[j]}, p={pvalue}")
spread = S1 - S2
zscore = (spread - spread.rolling(21, min_periods=21).mean()) / spread.rolling(21, min_periods=21).std()

mask = np.tril(np.ones_like(pvalue_matrix, dtype=bool)) | (pvalue_matrix >= 0.10)
plt.figure(figsize=(8, 6))
sns.heatmap(
    pvalue_matrix,
    mask=mask,
    xticklabels=tickers,
    yticklabels=tickers,
    cmap="RdYlGn_r",
    annot=True,
    fmt=".3f",
    cbar=True,
    vmin=0,
    vmax=1,
)
plt.title("Cointegration Test p-value Matrix")
plt.show()

divergence_points = {}

for t1, t2 in pairs:  # pairs already filtered with pval < 0.10
    S1 = df[t1]
    S2 = df[t2]

    # Spread
    spread = S1 - S2

    # Rolling z-score
    mean = spread.rolling(21).mean()
    std = spread.rolling(21).std()
    zscore = (spread - mean) / std

    # Divergence thresholds
    upper = zscore > 2
    lower = zscore < -2

    divergence = df.index[upper | lower]

    divergence_points[(t1, t2)] = divergence

    print(f"\n{t1} vs {t2} divergence dates:")
    print(divergence[:10])  # print first few

plt.figure(figsize=(10,5))
plt.plot(zscore)
plt.axhline(2, linestyle="--")
plt.axhline(-2, linestyle="--")
plt.axhline(0)
plt.title(f"Z-score of spread: {t1} vs {t2}")
plt.show()
