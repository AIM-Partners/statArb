"""
Key things to improve:
1. grid test model parameters
2. idle with treasury bills or index fund instead of cash
3. Add some capacity to adapt to prefer winning pairs and avoid losing pairs
4. add transaction costs
5. Add a max drawdown: peak portfolio value is way higher than ending value
6. make sure there is cash to cover short positions/short logic is correct in general
7. Add data visualization
8. break up into smaller functions
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

"""
These are the tickers and model paramters, can vary these/test other ones int he future
"""

@dataclass
class Config:
    tickers: list = field(default_factory=lambda: [
        "XOM", "CVX", "COP", "SLB", "MPC",
        "JPM", "BAC", "WFC", "GS", "MS",
        "KO", "PEP", "MDLZ", "GIS", "CPB",
        "HD", "LOW", "TGT", "WMT", "COST",
        "MSFT", "GOOGL", "META", "AAPL", "INTC",
        "JNJ", "PFE", "MRK", "ABT", "BMY",
        "HON", "MMM", "GE", "CAT", "DE",
        "NEE", "DUK", "SO", "AEP", "EXC",
        "KO", "PEP", "F", "GM", "SPY", "QQQ"
    ])

    start_date: str = "2019-01-01"
    end_date: str = "2023-01-01"
    train_ratio: float = 0.60

    coint_pvalue: float = 0.05
    min_half_life: float = 5.0
    max_half_life: float = 120.0
    max_pairs: int = 20

    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    stop_z_score: float = 3.5
    z_window: int = 60

    capital: float = 100_000.0
    trade_pct: float = 0.20
    max_open_positions: int = 5

cfg = Config()

"""
function to download price data from yfinance
"""
def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    print(f"\nDownloading price data for {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    print(f"{len(prices.columns)} tickers loaded, {len(prices)} trading days")
    return prices


"""
Below are some functions that are actually used to compute deviations in correlated stocks
"""
def half_life(spread: pd.Series) -> float:
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lag, delta = lag.align(delta, join="inner")

    if len(lag) < 2:
        return np.inf

    X = sm.add_constant(lag)
    try:
        lam = sm.OLS(delta, X).fit().params.iloc[1]
    except Exception:
        return np.inf

    if not np.isfinite(lam) or lam >= 0:
        return np.inf

    hl = -np.log(2) / lam
    return float(hl) if np.isfinite(hl) else np.inf


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    try:
        return float(sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1])
    except Exception:
        return np.nan

def zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (series - mu) / sig.replace(0, np.nan)


@dataclass
class PairStats:
    ticker1: str
    ticker2: str
    pvalue: float
    beta: float
    half_life: float

"""
This function keeps the pairs that re most closely cointegrated
"""
def find_cointegrated_pairs(prices: pd.DataFrame, cfg: Config) -> list[PairStats]:
    tickers = list(prices.columns)
    total = len(tickers) * (len(tickers) - 1) // 2
    print(f"\nTesting {total} pairs for cointegration …")

    pairs: list[PairStats] = []

    for t1, t2 in combinations(tickers, 2):
        y = prices[t1]
        x = prices[t2]

        try:
            _, pv, _ = coint(y, x)
        except Exception:
            continue
        if not np.isfinite(pv) or pv > cfg.coint_pvalue:
            continue
        beta = hedge_ratio(y, x)
        if not np.isfinite(beta) or beta <= 0:
            continue
        spread = y - beta * x
        hl = half_life(spread)
        if not (cfg.min_half_life <= hl <= cfg.max_half_life):
            continue
        pairs.append(PairStats(t1, t2, float(pv), float(beta), float(hl)))
    pairs.sort(key=lambda p: p.pvalue)
    keep = min(cfg.max_pairs, len(pairs))
    print(f" {len(pairs)} tradeable pairs found (keeping top {keep})")
    return pairs[:keep]


"""
Dataclasses representing either a positiont that is currently open, or a position that is currently closed
"""
@dataclass
class OpenPosition:
    pair_idx: int
    direction: int          # keeps track of long and short
    open_date: object

    shares_y: float         
    shares_x: float         

    open_y: float
    open_x: float

    open_z: float        #z score when this position was opened
    notional: float      #dollar value allocated for this trade


@dataclass
class ClosedTrade:
    pair: str
    direction: int
    open_date: object
    close_date: object
    open_z: float
    close_z: float
    pnl: float
    exit_reason: str


def run_portfolio_backtest(pairs, prices, test_start, cfg):
    test_idx = prices.index >= test_start
    dates = prices.index[test_idx]

    if len(dates) == 0:
        raise ValueError("No test-period dates found.")

    signals = {}
    for i, p in enumerate(pairs):
        spread = prices[p.ticker1] - p.beta * prices[p.ticker2]
        signals[i] = zscore(spread, cfg.z_window).shift(1)

    cash = cfg.capital
    open_pos: list[OpenPosition] = []
    closed_trades: list[ClosedTrade] = []

    start_loc = prices.index.get_indexer([test_start])[0]
    if start_loc > 0:
        equity_dates = [prices.index[start_loc - 1]]
    else:
        equity_dates = [test_start]
    equity_vals = [cfg.capital]

    def position_value(pos: OpenPosition, date) -> float:
        p = pairs[pos.pair_idx]
        py = prices[p.ticker1].get(date, np.nan)
        px = prices[p.ticker2].get(date, np.nan)
        if pd.isna(py) or pd.isna(px):
            return np.nan
        return pos.shares_y * py + pos.shares_x * px

    

    for date in dates:
        
        still_open = []
        for pos in open_pos:
            p = pairs[pos.pair_idx]

            py = prices[p.ticker1].get(date, np.nan)
            px = prices[p.ticker2].get(date, np.nan)
            sig = signals[pos.pair_idx].get(date, np.nan)

            if pd.isna(py) or pd.isna(px) or pd.isna(sig):
                still_open.append(pos)
                continue

            exit_reason = None
            if abs(sig) < cfg.exit_z_score:
                exit_reason = "mean_revert"
            elif (pos.direction == 1 and sig > cfg.stop_z_score) or (pos.direction == -1 and sig < -cfg.stop_z_score):
                exit_reason = "stop_loss"

            #close the position
            if exit_reason:
                open_value = pos.shares_y * pos.open_y + pos.shares_x * pos.open_x
                close_value = pos.shares_y * py + pos.shares_x * px
                trade_pnl = close_value - open_value
                cash += close_value

                closed_trades.append(
                    ClosedTrade(
                    pair=f"{p.ticker1}/{p.ticker2}", direction=pos.direction,
                    open_date=pos.open_date, close_date=date,
                    open_z=pos.open_z, close_z=float(sig),
                    pnl=float(trade_pnl), exit_reason=exit_reason,
                    )
                )
            else:
                still_open.append(pos)

        open_pos = still_open

        #Entry logic: rank by strongest signal, then fill up to max_open_positions
        active_pairs = {pos.pair_idx for pos in open_pos}
        candidates = []

        for i, p in enumerate(pairs):
            if i in active_pairs:
                continue

            sig = signals[i].get(date, np.nan)
            if pd.isna(sig):
                continue

            direction = 0
            if sig > cfg.entry_z_score:
                direction = -1
            elif sig < -cfg.entry_z_score:
                direction = 1
            else:
                continue

            candidates.append((abs(sig), i, direction, float(sig)))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, i, direction, sig in candidates:
            if len(open_pos) >= cfg.max_open_positions:
                break

            p = pairs[i]
            py = prices[p.ticker1].get(date, np.nan)
            px = prices[p.ticker2].get(date, np.nan)

            if pd.isna(py) or pd.isna(px):
                continue
            
            #notional = cfg.capital * cfg.trade_pct
            notional = cash * cfg.trade_pct

            denom = py + abs(p.beta) * px  #computing the number of shares of y to buy based on the computed hedge ratio
            if denom <= 1e-8:
                continue

            shares_y = notional / denom  
            shares_x = abs(p.beta) * shares_y

            if direction == 1:   # long spread: long y, short x
                pos_y = shares_y
                pos_x = -shares_x
            else:                # short spread: short y, long x
                pos_y = -shares_y
                pos_x = shares_x

            #immediate cash gained in a short position
            entry_value = pos_y * py + pos_x * px
            entry_cashflow = -entry_value

            # Cash cannot be negative
            if cash + entry_cashflow < 0:
                continue
            cash += entry_cashflow
            open_pos.append(OpenPosition( 
                pair_idx=i, direction=direction,
                open_date=date, shares_y=pos_y,
                shares_x=pos_x, open_y=py,
                open_x=px, open_z=sig,
                notional=notional,
                )
            )

        mtm = 0.0
        for pos in open_pos:
            mv = position_value(pos, date)
            if np.isfinite(mv):
                mtm += mv

        equity = cash + mtm
        equity_dates.append(date)
        equity_vals.append(equity)
        print(f"{date.date()} | Cash=${cash:.2f} | MTM=${mtm:.2f}")

    equity_s = pd.Series(equity_vals, index=equity_dates)

    pair_names = [f"{p.ticker1}/{p.ticker2}" for p in pairs]
    return portfolio_metrics(equity_s, closed_trades, pair_names)


def portfolio_metrics(equity: pd.Series, trades: list[ClosedTrade], pair_names: list[str]) -> dict:
    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_periods = max(len(equity) - 1, 1)

    if equity.iloc[0] > 0 and equity.iloc[-1] > 0:
        ann_ret = ((equity.iloc[-1] / equity.iloc[0]) ** (252 / n_periods) - 1) * 100
    else:
        ann_ret = np.nan

    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    roll_max = equity.cummax()
    max_dd = ((equity - roll_max) / roll_max).min() * 100

    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / max(len(trades), 1) * 100

    pair_stats = {}
    for name in pair_names:
        pt = [t for t in trades if t.pair == name]
        if not pt:
            continue
        pnls = [t.pnl for t in pt]
        pair_stats[name] = dict(
            n_trades=len(pt),
            win_rate=len([x for x in pnls if x > 0]) / len(pt) * 100,
            total_pnl=float(sum(pnls)),
            avg_pnl=float(np.mean(pnls)),
        )

    return dict(
        equity=equity,
        total_ret=float(total_ret),
        ann_ret=float(ann_ret) if np.isfinite(ann_ret) else np.nan,
        sharpe=float(sharpe),
        max_dd=float(max_dd),
        n_trades=len(trades),
        win_rate=float(win_rate),
        trades=trades,
        pair_stats=pair_stats,
    )


def main():
    prices = download_prices(cfg.tickers, cfg.start_date, cfg.end_date)
    if prices.empty or len(prices.columns) < 2:
        print("Not enough data.")
        sys.exit(1)

    split = int(len(prices) * cfg.train_ratio)
    if split <= 0 or split >= len(prices):
        print("Bad train/test split.")
        sys.exit(1)

    train = prices.iloc[:split]
    test_start = prices.index[split]

    print(f"\nTrain : {prices.index[0].date()} to {prices.index[split - 1].date()}")
    print(f"    Test  : {test_start.date()} to {prices.index[-1].date()}")

    pairs = find_cointegrated_pairs(train, cfg)
    if not pairs:
        print(" No cointegrated pairs found.")
        sys.exit(1)

    print(f"\nRunning portfolio backtest ({len(pairs)} pairs) …")
    result = run_portfolio_backtest(pairs, prices, test_start, cfg)

    print(f"\nPortfolio")
    print(f"    Total Return : {result['total_ret']:+.1f}%")
    print(f"    Ann. Return  : {result['ann_ret']:+.1f}%")
    print(f"    Sharpe Ratio : {result['sharpe']:.2f}")
    print(f"    Max Drawdown : {result['max_dd']:.1f}%")
    print(f"    Total Trades : {result['n_trades']}")
    print(f"    Win Rate     : {result['win_rate']:.0f}%")

    print("\n    Per-pair breakdown:")
    for name, ps in sorted(result["pair_stats"].items(), key=lambda kv: -kv[1]["total_pnl"]):
        flag = "Winning" if ps["total_pnl"] >= 0 else "Losing "
        print(f"  {flag}  {name:25s}  " f"P&L=${ps['total_pnl']:+8,.0f}  " f"Trades={ps['n_trades']:3d}  "f"WinRate={ps['win_rate']:.0f}%")

    print("\n Done.\n")

if __name__ == "__main__":
    main()