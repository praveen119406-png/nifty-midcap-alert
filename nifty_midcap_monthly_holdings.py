"""
Nifty Midcap 150 — Monthly Holdings & Exit Report
Shows monthly stock holdings, exits, and quantities
"""

import csv
import os
import sys
from datetime import datetime, timedelta
from collections import OrderedDict

import pandas as pd
import numpy as np
import yfinance as yf


CSV_PATH = r"C:\Users\prave\Downloads\ind_niftymidcap150list.csv"
INITIAL_CAPITAL = 100000
TOP_N = 5
RANK_EXIT_THRESHOLD = 10
LOOKBACK = 252
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
        print(f"[{i}/{total}] {sym}...", end="", flush=True)
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
                    print(f" OK ({len(closes)}d)")
                else:
                    print(f" skip ({len(closes)}d)")
            else:
                print(" no data")
        except Exception:
            print(" error")
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


def get_price(all_data, symbol, date):
    prices = all_data.get(symbol)
    if prices is None:
        return None
    available = prices[prices.index <= date]
    if available.empty:
        return None
    return float(available.iloc[-1])


def run_backtest(all_data, backtest_start, backtest_end):
    trading_days = sorted(set().union(*(prices.index for prices in all_data.values())))
    trading_days = [d for d in trading_days if backtest_start <= d <= backtest_end]

    portfolio = {}
    cash = float(INITIAL_CAPITAL)
    monthly_reports = []
    last_rebalance_month = None

    for date in trading_days:
        current_month = (date.year, date.month)

        if last_rebalance_month is None or current_month != last_rebalance_month:
            last_rebalance_month = current_month

            rankings = rank_stocks(all_data, date)
            rank_dict = {sym: i + 1 for i, (sym, _) in enumerate(rankings)}
            top_symbols = [sym for sym, _ in rankings[:TOP_N]]

            exited_stocks = []
            for sym in list(portfolio.keys()):
                rank = rank_dict.get(sym, 999)
                if rank > RANK_EXIT_THRESHOLD:
                    price = get_price(all_data, sym, date)
                    if price and portfolio[sym]["shares"] > 0:
                        pos = portfolio[sym]
                        proceeds = pos["shares"] * price
                        entry_price = pos["entry_price"]
                        pnl = (price - entry_price) / entry_price * 100
                        exited_stocks.append({
                            "symbol": sym,
                            "exit_price": round(price, 2),
                            "entry_price": round(entry_price, 2),
                            "shares": pos["shares"],
                            "pnl_pct": round(pnl, 2),
                            "proceeds": round(proceeds, 2),
                        })
                        cash += proceeds
                    del portfolio[sym]

            new_buys = []
            empty_slots = TOP_N - len(portfolio)
            if empty_slots > 0:
                buys = [s for s in top_symbols if s not in portfolio][:empty_slots]
                if buys:
                    alloc_per_stock = cash / len(buys)
                    for sym in buys:
                        price = get_price(all_data, sym, date)
                        if price and price > 0:
                            shares = int(alloc_per_stock / price)
                            if shares > 0:
                                cost = shares * price
                                portfolio[sym] = {"shares": shares, "entry_price": price}
                                new_buys.append({
                                    "symbol": sym,
                                    "buy_price": round(price, 2),
                                    "shares": shares,
                                    "investment": round(cost, 2),
                                    "rank": rank_dict.get(sym, 0),
                                })
                                cash -= cost

            total_value = cash
            holdings = []
            for sym, pos in portfolio.items():
                price = get_price(all_data, sym, date)
                if price:
                    mkt_val = pos["shares"] * price
                    total_value += mkt_val
                    pnl = (price - pos["entry_price"]) / pos["entry_price"] * 100
                    holdings.append({
                        "symbol": sym,
                        "shares": pos["shares"],
                        "entry_price": round(pos["entry_price"], 2),
                        "current_price": round(price, 2),
                        "market_value": round(mkt_val, 2),
                        "pnl_pct": round(pnl, 2),
                        "rank": rank_dict.get(sym, 0),
                    })

            monthly_reports.append({
                "date": date,
                "portfolio_value": round(total_value, 2),
                "cash": round(cash, 2),
                "holdings": sorted(holdings, key=lambda x: x["market_value"], reverse=True),
                "new_buys": new_buys,
                "exited": exited_stocks,
            })

    return monthly_reports


def main():
    symbols = load_stock_list(CSV_PATH)
    print(f"Loaded {len(symbols)} stocks\n")

    print("Downloading 5+ years data...")
    all_data = download_all_data(symbols, BACKTEST_YEARS + 1)
    print(f"\nGot data for {len(all_data)} stocks\n")

    backtest_end = pd.Timestamp(datetime.now().date())
    backtest_start = backtest_end - timedelta(days=BACKTEST_YEARS * 365)

    print(f"Backtesting: {backtest_start.date()} to {backtest_end.date()}\n")

    monthly_reports = run_backtest(all_data, backtest_start, backtest_end)

    lines = []
    lines.append("=" * 100)
    lines.append("NIFTY MIDCAP 150 MOMENTUM STRATEGY - MONTHLY HOLDINGS & EXIT REPORT")
    lines.append(f"Period: {backtest_start.date()} to {backtest_end.date()}")
    lines.append(f"Capital: Rs.{INITIAL_CAPITAL:,} | Top {TOP_N} Stocks | Exit if Rank > {RANK_EXIT_THRESHOLD}")
    lines.append("=" * 100)

    for report in monthly_reports:
        date_str = report["date"].strftime("%b %Y")
        lines.append(f"\n{'-' * 100}")
        lines.append(f"  {date_str}  |  Portfolio: Rs.{report['portfolio_value']:,.2f}  |  Cash: Rs.{report['cash']:,.2f}")
        lines.append(f"{'-' * 100}")

        if report["holdings"]:
            lines.append(f"  {'HOLDINGS':^90}")
            lines.append(f"  {'Symbol':<12} {'Shares':>10} {'Entry':>10} {'Current':>10} {'Value':>12} {'P&L%':>8} {'Rank':>6}")
            lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8} {'-'*6}")
            for h in report["holdings"]:
                pnl_str = f"{h['pnl_pct']:+.1f}%"
                lines.append(
                    f"  {h['symbol']:<12} {h['shares']:>10d} {h['entry_price']:>10.2f} "
                    f"{h['current_price']:>10.2f} {h['market_value']:>12.2f} {pnl_str:>8} {h['rank']:>6}"
                )

        if report["new_buys"]:
            lines.append(f"\n  NEW BUYS:")
            for b in report["new_buys"]:
                lines.append(f"    + {b['symbol']:<12} | Price: Rs.{b['buy_price']:>10.2f} | Shares: {b['shares']:d} | Invested: Rs.{b['investment']:,.2f} | Rank: {b['rank']}")

        if report["exited"]:
            lines.append(f"\n  EXITED:")
            for e in report["exited"]:
                lines.append(f"    - {e['symbol']:<12} | Entry: Rs.{e['entry_price']:>10.2f} | Exit: Rs.{e['exit_price']:>10.2f} | Shares: {e['shares']:d} | P&L: {e['pnl_pct']:+.1f}% | Proceeds: Rs.{e['proceeds']:,.2f}")

        lines.append("")

    report_text = "\n".join(lines)

    report_path = r"C:\Users\prave\Downloads\OPENCODE\monthly_holdings_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    sys.stdout.reconfigure(encoding="utf-8")
    print(report_text)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
