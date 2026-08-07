"""
Automated monthly rebalance monitor for the Nifty Midcap 150 momentum strategy.
Run daily via GitHub Actions (or cron); it performs the strategy's monthly
check on the first trading day of each month and sends the result to Telegram.

Usage:
    python automate_strategy.py --run         # normal daily run
    python automate_strategy.py --force       # run the check now regardless of date
    python automate_strategy.py --dry-run     # print message instead of sending
    python automate_strategy.py --test        # send a simple Telegram test message

Setup:
    1. Message @BotFather on Telegram -> /newbot -> copy the HTTP API token.
    2. Message your bot once, then GET https://api.telegram.org/bot<TOKEN>/getUpdates
       to find your chat_id, or ask @userinfobot.
    3. Configure the token/chat_id via env vars TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
       (GitHub Actions secrets) or fill in alert_config.json.
    4. Schedule daily runs (.github/workflows/momentum_alert.yml or cron below).

# crontab -e  (runs daily 18:45 IST; the monthly check fires on the first
#              trading day of the new month)
45 18 * * *  cd /home/user/momentum && python3 automate_strategy.py --run >> alert.log 2>&1
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

import backtest_momentum as bt

CONFIG_FILE = "alert_config.json"
STATE_FILE = "portfolio_state.json"

POSITIONS = bt.POSITIONS
RANK_THRESHOLD = bt.RANK_THRESHOLD


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def get_prices():
    symbols = bt.load_constituents(bt.CONST_FILE)
    prices = bt.load_prices(symbols)
    return {t: s for t, s in prices.items() if len(s) > bt.LOOKBACK_DAYS + 250}


def month_end_as_of(prices):
    """Last trading day of the most recent completed calendar month.

    The strategy rebalances on end-of-month closing prices only, so the
    ranking reference date is always the final trading day of the last
    completed month (the month before the current one), regardless of when
    the workflow happens to run. Intra-month prices are never used for
    ranking.
    """
    all_dates = sorted({pd.Timestamp(d) for s in prices.values() for d in s.index})
    # Most recent completed calendar month = the month before today. Any
    # data dated inside the current month is intra-month and must be ignored.
    ref = (pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1))
    completed = [d for d in all_dates if d <= ref]
    if not completed:
        completed = all_dates
    months = {}
    for d in completed:
        key = (d.year, d.month)
        months[key] = d if d > months.get(key, pd.Timestamp.min) else months[key]
    return months[max(months)]


def get_ranks(prices, as_of):
    mom = {}
    for t in prices:
        if t == bt.BENCHMARK:
            continue
        r = bt.one_year_return(prices[t], as_of)
        if r is not None and np.isfinite(r):
            mom[t] = r
    ranked = sorted(mom.items(), key=lambda kv: kv[1], reverse=True)
    rank = {t: i + 1 for i, (t, _) in enumerate(ranked)}
    return ranked, rank


def short(ticker):
    return ticker.replace(".NS", "")


def fmt_rs(x):
    return f"Rs {x:,.0f}"


def price_at(prices, t, as_of):
    return prices[t][prices[t].index <= as_of].iloc[-1]


def plan_rebalance(state, prices, ranked, rank, as_of):
    """Work out sells/buys without mutating the state. Returns (triggered, sells, buys)."""
    holdings = state["holdings"]
    cash = state["cash"]
    sells = [t for t in holdings if rank.get(t, RANK_THRESHOLD + 1) > RANK_THRESHOLD]
    triggered = bool(sells)
    buys = []  # (ticker, shares, px, rank)
    if triggered:
        pooled = cash + sum(
            holdings[t]["shares"] * price_at(prices, t, as_of) for t in sells
        )
        kept = {t for t in holdings if t not in sells}
        budget = pooled / len(sells)
        for t, _ in ranked:
            if len(kept) + len(buys) >= POSITIONS:
                break
            if t in holdings:
                continue
            px = price_at(prices, t, as_of)
            b = min(budget, pooled)
            shares = int(b // px)
            if shares == 0 and px <= pooled:
                shares = 1
            if shares > 0 and shares * px <= pooled:
                buys.append((t, shares, px, rank.get(t)))
                pooled -= shares * px
    return triggered, sells, buys


def apply_rebalance(state, prices, sells, buys, as_of):
    """Mutate the state to reflect the rebalance plan (advisory paper-tracking).

    Sells are removed at their as_of price (proceeds returned to cash), then
    buys are added at the computed share/price. Called only when a rebalance
    was triggered, after the Telegram message has been sent.
    """
    holdings = state["holdings"]
    cash = state["cash"]
    for t in sells:
        cash += holdings[t]["shares"] * price_at(prices, t, as_of)
        del holdings[t]
    for t, shares, px, _rk in buys:
        holdings[t] = {"shares": int(shares), "avg_price": float(px)}
        cash -= shares * px
    state["holdings"] = holdings
    state["cash"] = round(float(cash), 2)
    return state


def build_message(state, prices, ranked, rank, as_of, triggered, sells, buys):
    holdings = state["holdings"]
    cash = state["cash"]
    lines = []
    lines.append("NIFTY MIDCAP 150 MOMENTUM - MONTHLY CHECK")
    lines.append(f"As of {as_of.date()}")
    lines.append("")

    rows = []
    total_val = cash
    for t, info in holdings.items():
        px = price_at(prices, t, as_of)
        val = info["shares"] * px
        cost = info["shares"] * info["avg_price"]
        total_val += val
        rows.append((t, info["shares"], val, val - cost, rank.get(t, "-")))

    lines.append(f"PORTFOLIO VALUE: {fmt_rs(total_val)}   (idle cash {fmt_rs(cash)})")
    lines.append(f"P&L: {fmt_rs(total_val - state['initial_capital'])} "
                 f"({100 * (total_val / state['initial_capital'] - 1):+.2f}%)")
    lines.append("")
    lines.append("HOLDINGS:")
    for t, sh, val, pnl, rk in rows:
        lines.append(f" {short(t):<12} {sh:>4} sh  rank {str(rk):>2}  {fmt_rs(val):>10}  P&L {pnl:+,.0f}")
    lines.append("")

    if not triggered:
        lines.append("DECISION: NO REBALANCE - HOLD ALL")
    else:
        lines.append("DECISION: REBALANCE TRIGGERED (rank > 10)")
        for t in sells:
            lines.append(f"  SELL {short(t):<12} {holdings[t]['shares']} sh  rank {rank[t]}")
        for t, sh, px, rk in buys:
            lines.append(f"  BUY  {short(t):<12} rank {rk}  {sh} sh @ {fmt_rs(px)}")
    lines.append("")
    lines.append("TOP 10 MOMENTUM:")
    for i, (t, r) in enumerate(ranked[:10], 1):
        lines.append(f" {i:>2}. {short(t):<14} {r:+.1%}")
    return "\n".join(lines)


def run_check(force=False, dry_run=False, daily=False):
    prices = get_prices()
    as_of = month_end_as_of(prices)
    state = load_json(STATE_FILE)

    due = force or daily
    if not due and state.get("last_check"):
        ly, lm = map(int, state["last_check"].split("-"))
        due = (as_of.year, as_of.month) > (ly, lm)
    if not due:
        print(f"No monthly check due yet (month-end data {as_of.date()}, "
              f"last check {state['last_check']}). Nothing to do.")
        return

    ranked, rank = get_ranks(prices, as_of)
    triggered, sells, buys = plan_rebalance(state, prices, ranked, rank, as_of)
    msg = build_message(state, prices, ranked, rank, as_of, triggered, sells, buys)

    if dry_run:
        print(msg)
        print("\n[DRY RUN - nothing sent, state unchanged]")
        return

    # Send first: if Telegram fails, state is left unchanged so the check is
    # retried on the next run.
    send_telegram(msg)
    if triggered:
        apply_rebalance(state, prices, sells, buys, as_of)
        print("Rebalance applied to portfolio state (advisory paper-tracking).")
    state["last_check"] = f"{as_of.year}-{as_of.month:02d}"
    save_json(STATE_FILE, state)
    print("Telegram message sent, state updated.")


def send_test():
    send_telegram("Your Nifty Midcap 150 momentum monitor is configured and working.")
    print("Test message sent.")


def send_telegram(text):
    # Prefer environment variables (GitHub Actions secrets / local env), fall
    # back to alert_config.json.
    token = os.environ.get("TELEGRAM_TOKEN") or os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("CHAT_ID")
    if not token or not chat_id:
        cfg = load_json(CONFIG_FILE)
        token = cfg["bot_token"]
        chat_id = cfg["chat_id"]
    if "PASTE" in token:
        raise RuntimeError("Telegram not configured: set TELEGRAM_TOKEN/TELEGRAM_CHAT_ID env vars or edit alert_config.json.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=30)
    resp.raise_for_status()


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--test" in args:
        send_test()
    elif "--daily" in args:
        run_check(force="--force" in args, daily=True, dry_run="--dry-run" in args)
    elif "--dry-run" in args:
        run_check(force="--force" in args, dry_run=True)
    else:
        run_check(force="--force" in args, dry_run=False)
