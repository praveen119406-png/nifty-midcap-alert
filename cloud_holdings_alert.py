"""
Nifty Midcap 150 — Monthly Holdings Alert (GitHub Actions version)
Uses environment variables for secrets, fetches stock list from CSV in repo
"""

import csv
import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from math import floor
from io import StringIO

import pandas as pd
import numpy as np
import yfinance as yf


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CSV_URL = "https://raw.githubusercontent.com/NSE-India/Nifty-indices/main/nifty-indices/NIFTY%20MIDCAP%20150.csv"
STOCK_LIST_CSV = "ind_niftymidcap150list.csv"
PORTFOLIO_FILE = "portfolio.json"

TOP_N = 5
RANK_EXIT_THRESHOLD = 10
LOOKBACK = 252
INITIAL_CAPITAL = 100000


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not set, skipping send")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
        if not result.get("ok"):
            print(f"Telegram error: {result}")
        return result


def load_stock_list() -> list[tuple[str, str]]:
    if os.path.exists(STOCK_LIST_CSV):
        with open(STOCK_LIST_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            stocks = []
            for row in reader:
                sym = row.get("Symbol", "").strip()
                name = row.get("Company Name", "").strip()
                if sym:
                    stocks.append((sym, name))
            return stocks

    try:
        req = urllib.request.Request(CSV_URL)
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        stocks = []
        for row in reader:
            sym = row.get("Symbol", "").strip()
            name = row.get("Company Name", "").strip()
            if sym:
                stocks.append((sym, name))
        return stocks
    except Exception as e:
        print(f"Failed to load stock list: {e}")
        return []


def download_all_data(symbols: list[tuple[str, str]], years: int = 2) -> dict[str, pd.DataFrame]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + 60)
    all_data = {}
    total = len(symbols)
    for i, (sym, name) in enumerate(symbols, 1):
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


def calculate_momentum(prices: pd.Series) -> float | None:
    if len(prices) < LOOKBACK:
        return None
    current = float(prices.iloc[-1])
    past = float(prices.iloc[-LOOKBACK])
    if past == 0:
        return None
    return (current - past) / past


def rank_stocks(all_data: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
    scores = []
    for sym, prices in all_data.items():
        mom = calculate_momentum(prices)
        if mom is not None:
            scores.append((sym, mom))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def get_latest_price(prices: pd.Series) -> float:
    return float(prices.iloc[-1])


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_updated": None,
        "initial_capital": INITIAL_CAPITAL,
        "cash": float(INITIAL_CAPITAL),
        "holdings": {},
    }


def save_portfolio(portfolio: dict):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2)


def run_monthly():
    print("=" * 60)
    print("NIFTY MIDCAP 150 - MONTHLY HOLDINGS ALERT")
    print("=" * 60)

    stocks = load_stock_list()
    if not stocks:
        print("No stocks loaded, exiting")
        sys.exit(1)
    print(f"\nLoaded {len(stocks)} stocks")

    print("\nDownloading price data...")
    all_data = download_all_data(stocks)
    print(f"\nGot data for {len(all_data)} stocks")

    if len(all_data) < TOP_N:
        print(f"Not enough stocks with data ({len(all_data)} < {TOP_N})")
        sys.exit(1)

    rankings = rank_stocks(all_data)
    rank_dict = {sym: i + 1 for i, (sym, _) in enumerate(rankings)}
    top_symbols = [sym for sym, _ in rankings[:TOP_N]]

    portfolio = load_portfolio()
    current_holdings = portfolio["holdings"]

    holds = []
    exits = []
    buys = []

    for sym in list(current_holdings.keys()):
        rank = rank_dict.get(sym, 999)
        if sym in all_data:
            price = get_latest_price(all_data[sym])
        else:
            price = current_holdings[sym]["entry_price"]

        if rank > RANK_EXIT_THRESHOLD:
            entry_price = current_holdings[sym]["entry_price"]
            shares = current_holdings[sym]["shares"]
            pnl_pct = (price - entry_price) / entry_price * 100
            proceeds = shares * price
            exits.append({
                "symbol": sym,
                "shares": shares,
                "entry_price": round(entry_price, 2),
                "exit_price": round(price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "proceeds": round(proceeds, 2),
                "rank": rank,
            })
            portfolio["cash"] += proceeds
            del portfolio["holdings"][sym]
        else:
            holds.append({
                "symbol": sym,
                "shares": current_holdings[sym]["shares"],
                "entry_price": round(current_holdings[sym]["entry_price"], 2),
                "current_price": round(price, 2),
                "rank": rank,
            })

    empty_slots = TOP_N - len(portfolio["holdings"])
    if empty_slots > 0:
        new_syms = [s for s in top_symbols if s not in portfolio["holdings"]][:empty_slots]
        if new_syms:
            alloc_per_stock = portfolio["cash"] / len(new_syms)
            for sym in new_syms:
                if sym in all_data:
                    price = get_latest_price(all_data[sym])
                    if price > 0:
                        shares = floor(alloc_per_stock / price)
                        if shares > 0:
                            cost = shares * price
                            today = datetime.now().strftime("%Y-%m-%d")
                            portfolio["holdings"][sym] = {
                                "shares": shares,
                                "entry_price": round(price, 2),
                                "entry_date": today,
                            }
                            portfolio["cash"] -= cost
                            buys.append({
                                "symbol": sym,
                                "shares": shares,
                                "price": round(price, 2),
                                "cost": round(cost, 2),
                                "rank": rank_dict.get(sym, 0),
                            })

    portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_portfolio(portfolio)

    data_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg_lines = []
    msg_lines.append(f"<b>NIFTY MIDCAP 150 - MONTHLY REBALANCE</b>")
    msg_lines.append(f"Date: {data_date}")
    msg_lines.append(f"Stocks Analyzed: {len(all_data)}")
    msg_lines.append("")

    if holds:
        msg_lines.append(f"<b>HOLD ({len(holds)}):</b>")
        for h in holds:
            msg_lines.append(f"  {h['symbol']:<14} Rank:{h['rank']:<3} | {h['shares']} shares @ Rs.{h['entry_price']}")
        msg_lines.append("")

    if exits:
        msg_lines.append(f"<b>EXIT ({len(exits)}):</b>")
        for e in exits:
            pnl_str = f"{e['pnl_pct']:+.1f}%"
            msg_lines.append(f"  {e['symbol']:<14} Rank:{e['rank']:<3} | {e['shares']} shares | Entry:{e['entry_price']} -> Exit:{e['exit_price']} | P&L: {pnl_str}")
        msg_lines.append("")

    if buys:
        msg_lines.append(f"<b>BUY ({len(buys)}):</b>")
        for b in buys:
            msg_lines.append(f"  {b['symbol']:<14} Rank:{b['rank']:<3} | {b['shares']} shares @ Rs.{b['price']}")
        msg_lines.append("")

    total_value = portfolio["cash"]
    for sym in portfolio["holdings"]:
        if sym in all_data:
            total_value += portfolio["holdings"][sym]["shares"] * get_latest_price(all_data[sym])

    msg_lines.append(f"<b>Portfolio Value: Rs.{total_value:,.2f}</b>")
    msg_lines.append(f"Cash: Rs.{portfolio['cash']:,.2f}")
    msg_lines.append("")
    msg_lines.append("<i>Auto-generated by Nifty Midcap Alert Bot</i>")

    msg = "\n".join(msg_lines)

    print("\n--- Telegram Message ---")
    print(msg)
    print("\n--- Sending to Telegram ---")

    result = send_telegram(msg)
    if result.get("ok"):
        print("Message sent successfully!")
    else:
        print("Failed to send message.")

    print(f"\nPortfolio saved to: {PORTFOLIO_FILE}")


if __name__ == "__main__":
    run_monthly()
