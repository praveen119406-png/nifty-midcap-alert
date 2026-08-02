"""
Nifty Midcap 150 Momentum Strategy Backtest
--------------------------------------------
- Universe  : Nifty Midcap 150 constituents (NSE CSV)
- Signal    : 1-year (252 trading day) return momentum, ranked descending
- Portfolio : Top 5 stocks, Rs 20,000 capital each initially (Rs 100,000)
- Rebalance : Checked monthly. Rebalance only when a held stock's momentum
              rank falls below 10 (rank > 10). On rebalance only the stocks
              ranked worse than 10 are SOLD; the proceeds (profits reinvested)
              fund the best-ranked stocks not currently held. Portfolio stays 5.
- Data      : Yahoo Finance (daily adjusted close), cached locally.
- Costs     : ignored (per user request).
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_FILE = "price_data_cache.pkl"
CONST_FILE = "nifty_midcap150_constituents.csv"

DATA_START = "2019-01-01"
BACKTEST_START = "2021-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")  # always fetch through today

POSITIONS = 5
CAPITAL_PER_STOCK = 20000.0
LOOKBACK_DAYS = 252  # 1 year of trading days
RANK_THRESHOLD = 10  # rebalance when a held stock ranks worse than this

BENCHMARK = "NIFTYMIDCAP150.NS"  # Nifty Midcap 150 index on Yahoo


def load_constituents(path):
    df = pd.read_csv(path)
    symbols = [f"{s}.NS" for s in df["Symbol"].tolist()]
    return symbols


def load_prices(symbols):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as fh:
            cache = pickle.load(fh)
        missing = [s for s in symbols if s not in cache]
        if not missing and BENCHMARK in cache:
            print(f"Loaded price data from cache ({len(cache)} tickers).")
            return cache
        print(f"Cache incomplete ({len(missing)} missing). Refreshing...")
    else:
        cache = {}
        missing = symbols

    data = yf.download(
        missing + [BENCHMARK],
        start=DATA_START,
        end=END,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    for ticker in missing + [BENCHMARK]:
        try:
            sub = data[ticker]["Close"].dropna()
            sub.index = pd.to_datetime(sub.index)
            cache[ticker] = sub
        except Exception:
            pass

    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"Downloaded and cached data for {len(cache)} tickers.")
    return cache


def one_year_return(price_series, as_of):
    """Return over the LOOKBACK_DAYS trading days ending on/just before as_of."""
    hist = price_series[price_series.index <= as_of]
    if len(hist) < LOOKBACK_DAYS + 1:
        return np.nan
    past = hist.iloc[-LOOKBACK_DAYS - 1]
    now = hist.iloc[-1]
    return now / past - 1.0


def run_backtest(prices, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    # Aligned trading calendar = union of all data indexes
    all_idx = sorted(set().union(*[set(s.index) for s in prices.values()]))
    cal = pd.DatetimeIndex([d for d in all_idx if start <= d <= end])

    # Universe of valid stocks (enough history at backtest start)
    universe = [t for t in prices if t != BENCHMARK]

    # Monthly rebalance check dates: first trading day of each month
    check_dates = []
    seen = set()
    for d in cal:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            check_dates.append(d)

    cash = POSITIONS * CAPITAL_PER_STOCK
    holdings = {}  # ticker -> shares
    trades = []
    equity_curve = []

    for i, check in enumerate(check_dates):
        # Momentum ranks as of this check date
        mom = {}
        for t in universe:
            r = one_year_return(prices[t], check)
            if r is not None and np.isfinite(r):
                mom[t] = r
        if len(mom) < POSITIONS:
            continue
        ranked = sorted(mom.items(), key=lambda kv: kv[1], reverse=True)
        rank = {t: idx + 1 for idx, (t, _) in enumerate(ranked)}

        # Determine whether a rebalance is triggered this check
        held_now = set(holdings.keys())
        rebuild = holdings == {} or any(
            rank.get(t, RANK_THRESHOLD + 1) > RANK_THRESHOLD for t in held_now
        )

        def price_at(t):
            return prices[t][prices[t].index <= check].iloc[-1]

        if rebuild:
            # Sell ONLY positions whose momentum rank fell below 10; keep the
            # rest invested. Pool the proceeds and redeploy them (profits
            # reinvested) into the best-ranked stocks not currently held.
            for t in list(holdings.keys()):
                if rank.get(t, RANK_THRESHOLD + 1) > RANK_THRESHOLD:
                    px = price_at(t)
                    cash += holdings[t] * px
                    trades.append((check, "SELL", t, holdings[t], px))
                    del holdings[t]
            empty_slots = POSITIONS - len(holdings)
            if empty_slots > 0:
                budget = cash / empty_slots
                for t, _ in ranked:
                    if len(holdings) >= POSITIONS:
                        break
                    if t in holdings:
                        continue
                    px = price_at(t)
                    b = min(budget, cash)
                    shares = int(b // px)
                    if shares == 0 and px <= cash:
                        shares = 1
                    if shares > 0 and shares * px <= cash:
                        holdings[t] = shares
                        cash -= shares * px
                        trades.append((check, "BUY", t, shares, px))

        # Daily mark-to-market from this check until the next
        next_check = check_dates[i + 1] if i + 1 < len(check_dates) else None
        if next_check is None:
            window = cal[cal >= check]
        else:
            window = cal[(cal >= check) & (cal < next_check)]
        for day in window:
            value = cash
            for t, sh in holdings.items():
                hist = prices[t][prices[t].index <= day]
                if len(hist):
                    value += sh * hist.iloc[-1]
            equity_curve.append((day, value))

    eq = pd.DataFrame(equity_curve, columns=["Date", "Portfolio"]).set_index("Date")
    eq = eq[~eq.index.duplicated(keep="first")].sort_index()
    return eq, pd.DataFrame(trades, columns=["Date", "Action", "Ticker", "Shares", "Price"]), holdings


def benchmark_curve(prices):
    s = prices[BENCHMARK].copy()
    s = s[s.index >= BACKTEST_START]
    start_val = s.iloc[0]
    return (s / start_val) * (POSITIONS * CAPITAL_PER_STOCK)


def monthly_returns(series):
    m = series.resample("ME").last().ffill()
    return m.pct_change().dropna()


def stats(eq, bench_eq, trades):
    rows = {}

    def total(series):
        return series.iloc[-1] / series.iloc[0] - 1.0

    def cagr(series):
        years = (series.index[-1] - series.index[0]).days / 365.25
        return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1.0

    def max_dd(series):
        peak = series.cummax()
        return ((series / peak) - 1.0).min()

    def sharpe(series):
        r = monthly_returns(series)
        return (r.mean() * 12) / (r.std() * np.sqrt(12) + 1e-12)

    for name, series in (("Strategy", eq["Portfolio"]), ("Benchmark (Midcap150)", bench_eq)):
        rows[name + " | Total Return"] = f"{total(series):.2%}"
        rows[name + " | CAGR"] = f"{cagr(series):.2%}"
        rows[name + " | Max Drawdown"] = f"{max_dd(series):.2%}"
        rows[name + " | Sharpe (monthly, ann.)"] = f"{sharpe(series):.2f}"
    rows["Final Strategy Value"] = f"Rs {eq['Portfolio'].iloc[-1]:,.0f}"
    rows["Final Benchmark Value"] = f"Rs {bench_eq.iloc[-1]:,.0f}"
    rows["Rebalances (monthly checks with trade)"] = int(
        (trades["Date"].dt.to_period("M").nunique()) if len(trades) else 0
    )
    return rows


def main():
    symbols = load_constituents(CONST_FILE)
    print(f"Loaded {len(symbols)} constituents from {CONST_FILE}.")

    prices = load_prices(symbols)
    prices = {t: s for t, s in prices.items() if len(s) > LOOKBACK_DAYS + 250}
    if BENCHMARK not in prices:
        print("Benchmark data missing; cannot produce relative stats.")
        sys.exit(1)

    eq, trades, final_holdings = run_backtest(prices, BACKTEST_START, END)
    bench_eq = benchmark_curve(prices)

    eq.to_csv("equity_curve.csv")
    trades.to_csv("trades.csv", index=False)

    for k, v in stats(eq, bench_eq, trades).items():
        print(f"{k:<32} {v}")

    print(f"\nTotal trades: {len(trades)} | First: {trades['Date'].min().date()} | Last: {trades['Date'].max().date()}")

    # Current rank snapshot (most recent monthly check)
    latest = eq.index[-1]
    mom = {}
    for t in prices:
        if t == BENCHMARK:
            continue
        r = one_year_return(prices[t], latest)
        if r is not None and np.isfinite(r):
            mom[t] = r
    ranked = sorted(mom.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nTop-10 momentum stocks as of {latest.date()}:")
    for idx, (t, r) in enumerate(ranked[:10], 1):
        mark = " *Held" if t in final_holdings else ""
        print(f"  {idx:>2}. {t:<14} 1y return {r:>8.1%}{mark}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(eq.index, eq["Portfolio"], label="Strategy (Top-5 momentum)", lw=1.6)
        ax.plot(bench_eq.index, bench_eq, label="Nifty Midcap 150", lw=1.6, alpha=0.85)
        ax.set_title("Rs 100,000 momentum strategy vs Nifty Midcap 150")
        ax.set_ylabel("Portfolio value (Rs)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("equity_curve.png", dpi=140)
        print("\nSaved equity_curve.png")
    except Exception as exc:  # plotting is optional
        print(f"\nPlot failed: {exc}")


if __name__ == "__main__":
    main()
