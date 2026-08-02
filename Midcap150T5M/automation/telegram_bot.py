"""
Telegram command bot for the Nifty Midcap 150 momentum monitor.
Run locally (keeps running):  python telegram_bot.py

Commands (only answered for the authorized chat_id):
    /start /status /alert /check  -> current portfolio status + decision
    /test                          -> simple "pong" alive check

Credentials come from TELEGRAM_TOKEN / TELEGRAM_CHAT_ID env vars, or
fall back to alert_config.json.
"""
import time

import pandas as pd
import requests

import automate_strategy as am

POLL_TIMEOUT = 30


def get_config():
    token = os_environ_or_config("TELEGRAM_TOKEN", "BOT_TOKEN")
    chat_id = os_environ_or_config("TELEGRAM_CHAT_ID", "CHAT_ID")
    if "PASTE" in token:
        raise RuntimeError(
            "Telegram not configured: set TELEGRAM_TOKEN/TELEGRAM_CHAT_ID "
            "env vars or edit alert_config.json."
        )
    return token, int(chat_id)


def os_environ_or_config(*names):
    import os

    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    cfg = am.load_json(am.CONFIG_FILE)
    return cfg.get("bot_token" if "TOKEN" in names[0] else "chat_id", "")


def build_status():
    prices = am.get_prices()
    latest = max(pd.Timestamp(d) for s in prices.values() for d in s.index)
    state = am.load_json(am.STATE_FILE)
    ranked, rank = am.get_ranks(prices, latest)
    triggered, sells, buys = am.plan_rebalance(state, prices, ranked, rank, latest)
    return am.build_message(state, prices, ranked, rank, latest, triggered, sells, buys)


def main():
    token, chat_id = get_config()
    base = f"https://api.telegram.org/bot{token}"
    offset = 0
    print(f"Bot listening (authorized chat_id {chat_id}). Send /status to test.")
    while True:
        try:
            resp = requests.get(
                f"{base}/getUpdates",
                params={"timeout": POLL_TIMEOUT, "offset": offset},
                timeout=POLL_TIMEOUT + 10,
            )
            data = resp.json()
            if not data.get("ok"):
                time.sleep(5)
                continue
            for upd in data["result"]:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat = (msg.get("chat") or {}).get("id")
                text = (msg.get("text") or "").strip().lower()
                if chat != chat_id:
                    continue
                if text.startswith("/start") or text in ("/status", "/alert", "/check"):
                    print("Received status command, building reply...")
                    try:
                        reply = build_status()
                    except Exception as exc:
                        reply = f"Status check failed: {exc}"
                elif text.startswith("/test"):
                    reply = "pong - bot is alive"
                else:
                    continue
                requests.post(
                    f"{base}/sendMessage",
                    data={"chat_id": chat_id, "text": reply},
                    timeout=30,
                )
        except Exception as exc:
            print(f"Polling error: {exc}")
            time.sleep(10)


if __name__ == "__main__":
    main()
