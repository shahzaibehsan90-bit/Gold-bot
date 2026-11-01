"""
Backtest for the signal bot (last ~59 days of 15m data from Yahoo).

It replays the market bar by bar, runs the SAME bot logic (bot.run_market) at every
15-minute step, and then checks what would have happened to each signal:
  - limit order fills only if price touches Entry within the expiry window (8 bars)
  - after the fill, SL or TP, whichever is hit first (if both in the same bar -> SL, pessimistic)
  - no spread / commission / slippage is included (real results will be a bit worse)

Usage:  python backtest.py            (sends the result to Discord if DISCORD_WEBHOOK is set)
        MARKETS=XAUUSD,EURUSD python backtest.py
"""
import contextlib
import io
import os
import re
import sys

import pandas as pd

import bot

REAL_FETCH = bot.fetch      # keep the real downloader (bot.fetch gets replaced during the replay)

DAYS_15M = "59d"
EXPIRY = bot.P["expiry_bars"]
STRATS = [x.strip().upper() for x in os.getenv("BT_STRATS", "ICT,SB,JUDAS,BREAKER").split(",") if x.strip().upper() in bot.STRATEGIES]
MAX_HOLD = 96            # bars (24h) after fill, else counted as 'open'
NUM = r"([\d,]+\.?\d*)"
SIG_RE = re.compile(r"(BUY|SELL) \w+ LIMIT.*?Entry `" + NUM + r"` \| SL `" + NUM + r"` \| TP `" + NUM + r"`", re.S)
USD_RE = re.compile(r"Loss ~\$" + NUM + r" \| Profit ~\$" + NUM)


def num(s):
    return float(s.replace(",", ""))


def load(name):
    """Download history for one market (overridable in tests)."""
    tk = bot.MARKETS[name]["ticker"]
    m15 = REAL_FETCH(tk, "15m", DAYS_15M, 15)
    h1 = REAL_FETCH(tk, "60m", "59d", 60)
    d1 = REAL_FETCH(tk, "1d", "120d", 0, drop_forming=False)
    return m15, h1, d1


def simulate(side, entry, sl, tp, m15, t):
    """Return (result, bars_to_exit). result: 'unfilled' | 'win' | 'loss' | 'open'."""
    fut = m15[m15.index >= t]
    if fut.empty:
        return "open"
    H, L = fut["high"].values, fut["low"].values
    fill = None
    for i in range(min(EXPIRY, len(fut))):
        if (side == "BUY" and L[i] <= entry) or (side == "SELL" and H[i] >= entry):
            fill = i
            break
    if fill is None:
        return "unfilled"
    for j in range(fill, min(fill + MAX_HOLD, len(fut))):
        if side == "BUY":
            hit_sl, hit_tp = L[j] <= sl, H[j] >= tp
        else:
            hit_sl, hit_tp = H[j] >= sl, L[j] <= tp
        if hit_sl:                    # SL first when both are hit in the same bar
            return "loss"
        if hit_tp:
            return "win"
    return "open"


def backtest_market(name):
    """Returns {strategy_code: [trades]} (each strategy tested on its own)."""
    m15, h1, d1 = load(name)
    if m15 is None or h1 is None or d1 is None or len(m15) < 200:
        return None
    mk = bot.MARKETS[name]
    sent = []

    bot.send_discord = lambda text, img=None: sent.append(text)
    bot.make_chart = lambda *a, **k: None
    bot.save_state = lambda st: None
    bot.time.sleep = lambda x: None

    def fake_fetch(tk, interval, period, minutes, drop_forming=True):
        now = fake_fetch.now
        if interval == "15m":
            df, keep = m15, pd.Timedelta(days=mk["days"])
        elif interval == "60m":
            df, keep = h1, pd.Timedelta(days=30)
        else:
            df, keep = d1, pd.Timedelta(days=15)
        if interval == "1d":
            sel = df[(df.index <= now) & (df.index > now - keep)]
        else:
            end = df.index + pd.Timedelta(minutes=minutes)
            sel = df[(end <= now) & (df.index > now - keep)]
        return sel

    bot.fetch = fake_fetch
    steps = m15.index[160:] + pd.Timedelta(minutes=15)
    result = {}
    for code in STRATS:
        bot.ENABLED = [code]
        st = {}
        out = []
        for t in steps:
            fake_fetch.now = t
            sent.clear()
            try:
                with contextlib.redirect_stdout(io.StringIO()):      # keep the log quiet
                    bot.run_market(name, t, t.tz_convert(bot.LONDON), st)
            except Exception as ex:
                print(name, code, "err", t, ex)
                continue
            if not sent:
                continue
            msg = sent[0]
            m = SIG_RE.search(msg)
            u = USD_RE.search(msg)
            if not m or not u:
                continue
            side, e, sl, tp = m.group(1), num(m.group(2)), num(m.group(3)), num(m.group(4))
            loss_usd, win_usd = num(u.group(1)), num(u.group(2))
            res = simulate(side, e, sl, tp, m15, t)
            kz = bot.in_kill_zone(t.tz_convert(bot.LONDON))
            out.append(dict(t=t, side=side, res=res, loss=loss_usd, win=win_usd,
                            rr=abs(tp - e) / max(abs(e - sl), 1e-9), kz=kz, strat=code, name=name))
        result[code] = out
    result["_bars"] = len(m15)
    return result


def summarize(trades):
    n = len(trades)
    fills = [x for x in trades if x["res"] in ("win", "loss")]
    w = [x for x in fills if x["res"] == "win"]
    l = [x for x in fills if x["res"] == "loss"]
    net_r = sum(x["rr"] for x in w) - len(l)
    net_usd = sum(x["win"] for x in w) - sum(x["loss"] for x in l)
    wr = (100.0 * len(w) / len(fills)) if fills else 0.0
    return dict(signals=n, fills=len(fills), wins=len(w), losses=len(l),
                unfilled=sum(x["res"] == "unfilled" for x in trades),
                open=sum(x["res"] == "open" for x in trades),
                wr=wr, net_r=net_r, net_usd=net_usd)


def line(label, s):
    return (f"{label}: signal {s['signals']} | fill {s['fills']} | win {s['wins']} / loss {s['losses']} "
            f"| win-rate {s['wr']:.0f}% | net {s['net_r']:+.1f}R (~${s['net_usd']:+.0f} @0.01 lot)")


def cell(s):
    if not s["fills"]:
        return f"{s['signals']} sig/0 fill"
    return f"{s['wins']}W-{s['losses']}L {s['wr']:.0f}% {s['net_r']:+.1f}R"


def main():
    names = bot.ACTIVE
    by_strat = {c: [] for c in STRATS}
    rows = []
    for name in names:
        try:
            res = backtest_market(name)
        except Exception as ex:
            print(name, "backtest error:", ex)
            res = None
        if res is None:
            rows.append(f"{name}: data nahi mili")
            continue
        parts = []
        for c in STRATS:
            by_strat[c] += res[c]
            parts.append(f"{c} {cell(summarize(res[c]))}")
        rows.append(f"{name} ({res['_bars']} bars): " + " | ".join(parts))
        print(rows[-1])

    head = (f"📊 BACKTEST (pichle ~59 din; spread/commission ke baghair; ek candle mein SL+TP dono to SL pehle)\n"
            f"ALL_DAY={os.getenv('ALL_DAY', '0')} | MIN_SCORE={bot.P['min_score']} | har strategy alag test hui\n\n")
    tot = []
    for c in STRATS:
        tr = by_strat[c]
        label = bot.STRATEGIES[c][0]
        tot.append(line(f"**{label}**", summarize(tr)))
        tot.append("   kill zone: " + cell(summarize([x for x in tr if x["kz"]])) +
                   " | baqi ghante: " + cell(summarize([x for x in tr if not x["kz"]])))
    text = head + "\n".join(tot) + "\n\nPer market:\n" + "\n".join(rows) + \
        "\n\n⚠️ Sirf pichle 2 mahine ka nateeja hai, aage ki guarantee nahi. W=win, L=loss, R=risk ke multiple."
    print(text)

    hook = bot.WEBHOOK
    if hook:
        import requests
        for i in range(0, len(text), 1900):
            requests.post(hook, json={"content": text[i:i + 1900]}, timeout=30)


if __name__ == "__main__":
    main()
