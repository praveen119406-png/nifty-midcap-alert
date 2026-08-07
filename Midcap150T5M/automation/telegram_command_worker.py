"""
One-shot Telegram command poller, designed for GitHub Actions.
Called once per workflow run (scheduled every few minutes). It consumes
pending updates, answers commands for the authorized chat, and persists the
poll offset to telegram_offset.json so updates are not reprocessed.

Commands: /start /status /alert /check  -> portfolio status;  /top10 -> top 10 momentum;  /test -> pong
"""
import json
import os
import sys

import requests

import automate_strategy as am

OFFSET_FILE = "telegram_offset.json"
POLL_TIMEOUT = 2


def get_config():
    token = os.environ.get("TELEGRAM_TOKEN") or os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("CHAT_ID")
    if not token or not chat_id:
        cfg = am.load_json(am.CONFIG_FILE)
        token = cfg["bot_token"]
        chat_id = cfg["chat_id"]
    if "PASTE" in token:
        raise RuntimeError(
            "Telegram not configured: set TELEGRAM_TOKEN/TELEGRAM_CHAT_ID "
            "env vars or edit alert_config.json."
        )
    return token, int(chat_id)


def load_offset():
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh).get("offset", 0)
    return 0


def save_offset(offset):
    with open(OFFSET_FILE, "w", encoding="utf-8") as fh:
        json.dump({"offset": offset}, fh)


def build_status():
    prices = am.get_prices()
    as_of = am.month_end_as_of(prices)
    state = am.load_json(am.STATE_FILE)
    ranked, rank = am.get_ranks(prices, as_of)
    triggered, sells, buys = am.plan_rebalance(state, prices, ranked, rank, as_of)
    return am.build_message(state, prices, ranked, rank, as_of, triggered, sells, buys)


def build_top10():
    prices = am.get_prices()
    as_of = am.month_end_as_of(prices)
    ranked, _ = am.get_ranks(prices, as_of)
    lines = [f"TOP 10 NIFTY MIDCAP 150 MOMENTUM", f"As of {as_of.date()}", ""]
    for i, (t, r) in enumerate(ranked[:10], 1):
        lines.append(f"{i:>2}. {am.short(t):<14} {r:+.1%}")
    return "\n".join(lines)


def main():
    token, chat_id = get_config()
    base = f"https://api.telegram.org/bot{token}"
    offset = load_offset()
    resp = requests.get(
        f"{base}/getUpdates",
        params={"timeout": POLL_TIMEOUT, "offset": offset},
        timeout=POLL_TIMEOUT + 10,
    )
    data = resp.json()
    if not data.get("ok"):
        print("getUpdates failed:", data)
        sys.exit(1)
    updates = data.get("result", [])
    replied = 0
    new_offset = offset
    for upd in updates:
        new_offset = max(new_offset, upd["update_id"] + 1)
        msg = upd.get("message") or {}
        chat = (msg.get("chat") or {}).get("id")
        text = (msg.get("text") or "").strip().lower()
        if chat != chat_id:
            continue
        if text.startswith("/start") or text in ("/status", "/alert", "/check"):
            print("Building status reply...")
            try:
                reply = build_status()
            except Exception as exc:
                reply = f"Status check failed: {exc}"
        elif text.startswith("/top10"):
            print("Building top10 reply...")
            try:
                reply = build_top10()
            except Exception as exc:
                reply = f"Top10 check failed: {exc}"
        elif text.startswith("/test"):
            reply = "pong - bot is alive"
        else:
            continue
        requests.post(
            f"{base}/sendMessage",
            data={"chat_id": chat_id, "text": reply},
            timeout=30,
        )
        replied += 1
    save_offset(new_offset)
    print(f"Poll done. updates={len(updates)} replied={replied} offset={new_offset}")


if __name__ == "__main__":
    main()
