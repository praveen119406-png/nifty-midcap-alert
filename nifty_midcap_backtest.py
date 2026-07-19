"""
Nifty Midcap 150 — Momentum Strategy Backtest
Strategy: Buy top 5 stocks by 252-day return, rebalance monthly if rank > 10
"""

import csv
import os
import sys
from datetime import datetime, timedelta
from collections import OrderedDict

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CSV_PATH = r"C:\Users\prave\Downloads\ind_niftymidcap150list.csv"
INITIAL_CAPITAL = 100000
TOP_N = 5
RANK_EXIT_THRESHOLD = 10
LOOKBACK = 252  # trading days for momentum ranking
BACKTEST_YEARS = 5


def load_stock_list(csv_path: str) -> list[str]:
    symbols = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("Symbol", "").strip()
            if sym:
                symbols.append(sym)
    return symbols


def download_all_data(symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 60)

    all_data = {}
    total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        ticker = f"{sym}.NS"
        print(f"[{i}/{total}] Downloading {sym}...", end="", flush=True)
        try:
            data = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
            if data is not None and not data.empty:
                closes = data["Close"]
                if isinstance(closes, pd.DataFrame):
                    closes = closes.iloc[:, 0]
                closes = closes.dropna()
                if len(closes) > LOOKBACK:
                    all_data[sym] = closes
                    print(f" {len(closes)} days")
                else:
                    print(f" skipped ({len(closes)} days < {LOOKBACK})")
            else:
                print(" no data")
        except Exception as e:
            print(f" error: {e}")
    return all_data


def calculate_momentum(prices: pd.Series, as_of_date: pd.Timestamp) -> float | None:
    available = prices[prices.index <= as_of_date]
    if len(available) < LOOKBACK:
        return None
    current = float(available.iloc[-1])
    past = float(available.iloc[-LOOKBACK])
    if past == 0:
        return None
    return (current - past) / past


def rank_stocks(all_data: dict[str, pd.DataFrame], as_of_date: pd.Timestamp) -> list[tuple[str, float]]:
    scores = []
    for sym, prices in all_data.items():
        mom = calculate_momentum(prices, as_of_date)
        if mom is not None:
            scores.append((sym, mom))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def get_price(all_data: dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp) -> float | None:
    prices = all_data.get(symbol)
    if prices is None:
        return None
    available = prices[prices.index <= date]
    if available.empty:
        return None
    return float(available.iloc[-1])


def run_backtest(all_data: dict[str, pd.DataFrame], backtest_start: pd.Timestamp, backtest_end: pd.Timestamp):
    trading_days = sorted(
        set().union(*(prices.index for prices in all_data.values()))
    )
    trading_days = [d for d in trading_days if backtest_start <= d <= backtest_end]

    portfolio = {}  # symbol -> {"shares": float, "entry_price": float}
    cash = float(INITIAL_CAPITAL)
    total_value_history = []
    trade_log = []
    rebalance_dates = []
    last_rebalance_month = None

    for date in trading_days:
        current_month = (date.year, date.month)

        if last_rebalance_month is None or current_month != last_rebalance_month:
            last_rebalance_month = current_month
            rebalance_dates.append(date)

            rankings = rank_stocks(all_data, date)
            rank_dict = {sym: i + 1 for i, (sym, _) in enumerate(rankings)}
            top_symbols = [sym for sym, _ in rankings[:TOP_N]]

            sell_syms = []
            for sym in list(portfolio.keys()):
                rank = rank_dict.get(sym, 999)
                if rank > RANK_EXIT_THRESHOLD:
                    sell_syms.append(sym)

            for sym in sell_syms:
                price = get_price(all_data, sym, date)
                if price and portfolio[sym]["shares"] > 0:
                    proceeds = portfolio[sym]["shares"] * price
                    cash += proceeds
                    ret = (price - portfolio[sym]["entry_price"]) / portfolio[sym]["entry_price"] * 100
                    trade_log.append({
                        "date": date, "action": "SELL", "symbol": sym,
                        "price": price, "shares": portfolio[sym]["shares"],
                        "proceeds": proceeds, "return_pct": ret,
                    })
                del portfolio[sym]

            empty_slots = TOP_N - len(portfolio)
            if empty_slots > 0:
                new_buys = [s for s in top_symbols if s not in portfolio][:empty_slots]
                if new_buys:
                    alloc_per_stock = cash / len(new_buys)
                    for sym in new_buys:
                        price = get_price(all_data, sym, date)
                        if price and price > 0:
                            shares = int(alloc_per_stock / price)
                            if shares > 0:
                                cost = shares * price
                                portfolio[sym] = {"shares": shares, "entry_price": price}
                                cash -= cost
                                trade_log.append({
                                    "date": date, "action": "BUY", "symbol": sym,
                                    "price": price, "shares": shares,
                                    "proceeds": cost, "return_pct": 0,
                                })

        total_value = cash
        for sym, pos in portfolio.items():
            price = get_price(all_data, sym, date)
            if price:
                total_value += pos["shares"] * price
        total_value_history.append({"date": date, "value": total_value})

    return total_value_history, trade_log, rebalance_dates


def calculate_metrics(value_history: list[dict]) -> dict:
    df = pd.DataFrame(value_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    initial = df["value"].iloc[0]
    final = df["value"].iloc[-1]
    total_return = (final - initial) / initial * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    daily_returns = df["value"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    rolling_max = df["value"].cummax()
    drawdown = (df["value"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    return {
        "initial_capital": initial,
        "final_value": final,
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "volatility_pct": round(volatility, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "years": round(years, 1),
    }


def plot_results(value_history: list[dict], trade_log: list[dict], metrics: dict):
    df = pd.DataFrame(value_history)
    df["date"] = pd.to_datetime(df["date"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(df["date"], df["value"], linewidth=2, color="#2196F3", label="Portfolio")
    ax1.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.set_title(
        f"Nifty Midcap 150 Momentum Backtest\n"
        f"CAGR: {metrics['cagr_pct']}% | Total Return: {metrics['total_return_pct']}% | "
        f"Sharpe: {metrics['sharpe_ratio']} | Max DD: {metrics['max_drawdown_pct']}%",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value (Rs.)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    buy_dates = [t["date"] for t in trade_log if t["action"] == "BUY"]
    buy_vals = []
    for bd in buy_dates:
        row = df[df["date"] == bd]
        if not row.empty:
            buy_vals.append(row["value"].values[0])
    ax1.scatter(buy_dates, buy_vals, marker="^", color="green", s=50, zorder=5, label="Buy")

    sell_dates = [t["date"] for t in trade_log if t["action"] == "SELL"]
    sell_vals = []
    for sd in sell_dates:
        row = df[df["date"] == sd]
        if not row.empty:
            sell_vals.append(row["value"].values[0])
    ax1.scatter(sell_dates, sell_vals, marker="v", color="red", s=50, zorder=5, label="Sell")
    ax1.legend()

    ax2 = axes[1]
    drawdown = (df["value"] - df["value"].cummax()) / df["value"].cummax() * 100
    ax2.fill_between(df["date"], drawdown, 0, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    chart_path = r"C:\Users\prave\Downloads\OPENCODE\backtest_results.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {chart_path}")
    return chart_path


def main():
    symbols = load_stock_list(CSV_PATH)
    print(f"Loaded {len(symbols)} stocks from Nifty Midcap 150\n")

    print("Downloading historical data (5+ years)...")
    all_data = download_all_data(symbols, BACKTEST_YEARS + 1)
    print(f"\nGot data for {len(all_data)} stocks\n")

    backtest_end = pd.Timestamp(datetime.now().date())
    backtest_start = backtest_end - timedelta(days=BACKTEST_YEARS * 365)

    print(f"Backtesting: {backtest_start.date()} to {backtest_end.date()}")
    print(f"Initial Capital: Rs.{INITIAL_CAPITAL:,}")
    print(f"Strategy: Top {TOP_N} by {LOOKBACK}-day return, rebalance monthly if rank > {RANK_EXIT_THRESHOLD}\n")

    value_history, trade_log, rebalance_dates = run_backtest(all_data, backtest_start, backtest_end)

    metrics = calculate_metrics(value_history)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:          {backtest_start.date()} to {backtest_end.date()} ({metrics['years']} years)")
    print(f"  Initial Capital: Rs.{metrics['initial_capital']:,.2f}")
    print(f"  Final Value:     Rs.{metrics['final_value']:,.2f}")
    print(f"  Total Return:    {metrics['total_return_pct']:+.2f}%")
    print(f"  CAGR:            {metrics['cagr_pct']:+.2f}%")
    print(f"  Volatility:      {metrics['volatility_pct']:.2f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Rebalances:      {len(rebalance_dates)}")
    print(f"  Total Trades:    {len(trade_log)}")
    print("=" * 60)

    buys = [t for t in trade_log if t["action"] == "BUY"]
    sells = [t for t in trade_log if t["action"] == "SELL"]
    if sells:
        avg_sell_return = np.mean([t["return_pct"] for t in sells])
        win_trades = [t for t in sells if t["return_pct"] > 0]
        win_rate = len(win_trades) / len(sells) * 100
        print(f"\n  Avg Sell Return: {avg_sell_return:+.2f}%")
        print(f"  Win Rate:        {win_rate:.1f}% ({len(win_trades)}/{len(sells)})")

    chart_path = plot_results(value_history, trade_log, metrics)

    trade_df = pd.DataFrame(trade_log)
    trade_df.to_csv(r"C:\Users\prave\Downloads\OPENCODE\backtest_trades.csv", index=False)

    value_df = pd.DataFrame(value_history)
    value_df.to_csv(r"C:\Users\prave\Downloads\OPENCODE\backtest_value_history.csv", index=False)

    print(f"Trade log: backtest_trades.csv")
    print(f"Value history: backtest_value_history.csv")

    return metrics, chart_path


if __name__ == "__main__":
    main()
