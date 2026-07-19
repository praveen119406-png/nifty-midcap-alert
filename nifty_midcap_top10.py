"""
Nifty Midcap 150 — Multi-timeframe Returns (Corrected v2)
Uses raw prices (auto_adjust=False) for accurate price-based returns
"""

import csv
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def load_stock_list(csv_path: str) -> list[tuple[str, str]]:
    stocks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("Symbol", "").strip()
            name = row.get("Company Name", "").strip()
            if sym:
                stocks.append((sym, name))
    return stocks


def get_returns(symbol: str) -> dict | None:
    ticker = f"{symbol}.NS"
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1830)
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
        if data is None or data.empty:
            return None

        closes = data["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()

        if len(closes) < 2:
            return None

        last_date = closes.index[-1]
        last_price = float(closes.iloc[-1])

        periods = {
            "1W": 7,
            "1M": 30,
            "3M": 91,
            "1Y": 365,
            "3Y": 1095,
            "5Y": 1825,
        }

        results = {}
        for label, days in periods.items():
            target_date = last_date - timedelta(days=days)
            available = closes[closes.index <= target_date]
            if available.empty:
                results[label] = None
            else:
                first = float(available.iloc[-1])
                if first == 0:
                    results[label] = None
                else:
                    results[label] = round(((last_price - first) / first) * 100, 2)

        results["current_price"] = round(last_price, 2)
        return results
    except Exception:
        return None


def main():
    csv_path = r"C:\Users\prave\Downloads\ind_niftymidcap150list.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    stocks = load_stock_list(csv_path)
    print(f"Loaded {len(stocks)} stocks. Fetching data...\n")

    results = []
    total = len(stocks)
    for i, (sym, name) in enumerate(stocks, 1):
        print(f"[{i}/{total}] {sym}...", end="", flush=True)
        data = get_returns(sym)
        if data:
            row = {"symbol": sym, "name": name, **data}
            results.append(row)
            ret_1y = data.get("1Y")
            print(f" 1Y={ret_1y:+.2f}%" if ret_1y is not None else " done")
        else:
            print(" skipped")

    if not results:
        print("\nNo data retrieved.")
        sys.exit(1)

    df = pd.DataFrame(results)
    period_cols = ["1W", "1M", "3M", "1Y", "3Y", "5Y"]
    display_cols = ["symbol", "name", "current_price"] + period_cols

    df_sorted = df.sort_values("1Y", ascending=False, na_position="last").reset_index(drop=True)

    print("\n" + "=" * 130)
    print("NIFTY MIDCAP 150 — ALL STOCKS MULTI-TIMEFRAME RETURNS (%) — Sorted by 1Y Return")
    print("=" * 130)
    print(df_sorted[display_cols].to_string(index=False))
    print("=" * 130)
    print(f"\nData as of: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total stocks: {len(results)}/{total}")

    df_sorted[display_cols].to_csv("nifty_midcap_all_returns.csv", index=False)
    print("Saved to: nifty_midcap_all_returns.csv")


if __name__ == "__main__":
    main()
