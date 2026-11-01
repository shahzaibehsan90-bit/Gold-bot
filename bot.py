"""
ICT / SMC signal bot (XAUUSD + BTCUSD) -> Discord (webhook, chart image with Entry/SL/TP)

Concepts: HTF structure bias (BOS/HH-HL/LH-LL), liquidity sweep (PDH/PDL,
Asian range, swing), MSS/CHoCH, FVG, Order Block (+OB/FVG overlap entry),
premium/discount + OTE, London/NY kill zones, TP at next liquidity.

It only SENDS SIGNALS. You place the trade yourself in MT5 mobile (0.01 lot).
No profit guarantee. Test on demo first.

Usage:
  python bot.py           # run once (GitHub Actions calls this every 15 min)
  python bot.py --loop    # run forever (if you host it on a server)
  TEST=1 python bot.py    # send a test message + chart to Discord
  MARKETS=XAUUSD python bot.py   # only one market (default: XAUUSD,BTCUSD)
"""
import io
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from zoneinfo import ZoneInfo

WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
LONDON = ZoneInfo("Europe/London")
NEWYORK = ZoneInfo("America/New_York")
STATE_FILE = "state.json"

# US stocks / Nasdaq 100 trading windows, New York clock (hour as decimal):
# 09:30-12:00 (NY morning) and 13:30-15:30 (NY afternoon)
US_WINDOWS = ((9.5, 12.0), (13.5, 15.5))

# Per-market settings.
#  contract : units per 1.0 lot (gold 100 oz, forex 100000, crypto 1 coin on most MT5 brokers - check yours)
#  dec      : price decimals to show
#  usd      : approx USD value of 1 unit of the quote currency (only used for the max-loss safety filter)
#  max_loss_usd : skip the signal if SL loss at `lot` would be bigger than this
NOTE = "Price spot feed ka hai, MT5 se thora farq ho sakta hai. 'Feed price' ko MT5 ke price se milakar levels adjust karo."
CRYPTO_NOTE = NOTE + " (Crypto ka lot size broker ka check karo.)"


US_NOTE = NOTE + " (Index/stock ka lot size broker ka check karo, kai brokers 0.01 lot allow nahi karte.)"


# pip size per market (None = no pips, show only dollars)
PIP = {"XAUUSD": 0.1, "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
       "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001, "NZDUSD": 0.0001,
       "USDCHF": 0.0001, "USDCAD": 0.0001, "EURGBP": 0.0001}


def mkt(ticker, contract, dec, usd=1.0, weekend=False, note=NOTE, zone="fx", days=5):
    # zone "fx"  : London/NY kill zones (London clock)
    # zone "us"  : US stock session windows (New York clock)
    return dict(ticker=ticker, contract=contract, lot=0.01, max_loss_usd=10.0,
                weekend=weekend, kill_zone=True, dec=dec, usd=usd, note=note,
                zone=zone, days=days)


MARKETS = {
    "XAUUSD": mkt("XAUUSD=X|XAU=X|GC=F", 100, 2),
    "EURUSD": mkt("EURUSD=X", 100000, 5),
    "GBPUSD": mkt("GBPUSD=X", 100000, 5),
    "AUDUSD": mkt("AUDUSD=X", 100000, 5),
    "NZDUSD": mkt("NZDUSD=X", 100000, 5),
    "USDJPY": mkt("USDJPY=X", 100000, 3, usd=0.0065),
    "USDCHF": mkt("USDCHF=X", 100000, 5, usd=1.25),
    "USDCAD": mkt("USDCAD=X", 100000, 5, usd=0.72),
    "EURGBP": mkt("EURGBP=X", 100000, 5, usd=1.3),
    "EURJPY": mkt("EURJPY=X", 100000, 3, usd=0.0065),
    "GBPJPY": mkt("GBPJPY=X", 100000, 3, usd=0.0065),
    "BTCUSD": mkt("BTC-USD", 1, 2, weekend=True, note=CRYPTO_NOTE),
    "ETHUSD": mkt("ETH-USD", 1, 2, weekend=True, note=CRYPTO_NOTE),
    # Nasdaq 100 index + big US stocks (US session only)
    "NAS100": mkt("^NDX", 1, 2, note=US_NOTE, zone="us", days=10),
    "AAPL":   mkt("AAPL", 1, 2, note=US_NOTE, zone="us", days=10),
    "MSFT":   mkt("MSFT", 1, 2, note=US_NOTE, zone="us", days=10),
    "NVDA":   mkt("NVDA", 1, 2, note=US_NOTE, zone="us", days=10),
    "TSLA":   mkt("TSLA", 1, 2, note=US_NOTE, zone="us", days=10),
    "AMZN":   mkt("AMZN", 1, 2, note=US_NOTE, zone="us", days=10),
    "META":   mkt("META", 1, 2, note=US_NOTE, zone="us", days=10),
    "GOOGL":  mkt("GOOGL", 1, 2, note=US_NOTE, zone="us", days=10),
}
ALL = ",".join(MARKETS)
ACTIVE = [m.strip().upper() for m in os.getenv("MARKETS", ALL).split(",")]
ACTIVE = [m for m in ACTIVE if m in MARKETS]

P = dict(
    sweep_window=8, lookback=20, struct_bars=4, atr_period=14,
    min_disp_atr=0.8, sl_buf_atr=0.2, min_sl_atr=0.5, max_sl_atr=3.0,
    min_score=int(os.getenv("MIN_SCORE", "8")),
    require_discount=True,
    min_rr=1.5, default_rr=2.5, max_rr=5.0,
    range_bars=40, expiry_bars=8,
    london=(7, 11), ny=(12, 16), asian_end=7,   # kill zones (London clock), Asian range (UTC)
)


# ----------------------------------------------------------------- data
def series(a):
    """Newest-first array with a dummy at index 0, so index 1 = last closed bar."""
    return np.concatenate([[np.nan], np.asarray(a, dtype=float)[::-1]])


def fetch(ticker, interval, period, minutes, drop_forming=True):
    # ticker can be "A|B|C": backups are tried in order if the first has no data
    df = None
    for tk in str(ticker).split("|"):
        try:
            df = yf.download(tk, interval=interval, period=period,
                             progress=False, auto_adjust=False)
        except Exception as ex:
            print("download error", tk, ex); df = None
        if df is not None and not df.empty:
            print("data from", tk, len(df)); break
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close"]].dropna()
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    if drop_forming:
        now = pd.Timestamp.now(tz="UTC")
        while len(df) and df.index[-1] + pd.Timedelta(minutes=minutes) > now:
            df = df.iloc[:-1]
    return df


# ----------------------------------------------------------------- logic
def in_kill_zone(now_london):
    h = now_london.hour
    return (P["london"][0] <= h < P["london"][1]) or (P["ny"][0] <= h < P["ny"][1])


def zone_ok(mk, now_utc, now_london):
    """Is this market inside its trading window right now?"""
    if not mk["kill_zone"]:
        return True
    if mk.get("zone") == "us":
        ny = now_utc.tz_convert(NEWYORK)
        if ny.weekday() >= 5:
            return False
        t = ny.hour + ny.minute / 60.0
        return any(a <= t < b for a, b in US_WINDOWS)
    return in_kill_zone(now_london)


def bias_and_range(h1):
    """+1 bullish / -1 bearish / 0 none, plus dealing range for premium/discount."""
    if h1 is None or len(h1) < 40:
        return 0, None, None
    bh, bl = series(h1["high"]), series(h1["low"])
    n = len(bh)
    c1 = float(h1["close"].iloc[-1])
    rb = min(P["range_bars"], n - 4)
    rng_hi = float(np.nanmax(bh[1:rb + 1]))
    rng_lo = float(np.nanmin(bl[1:rb + 1]))

    sw_h, sw_l = [], []
    for i in range(3, n - 3):
        if len(sw_h) >= 2 and len(sw_l) >= 2:
            break
        if len(sw_h) < 2 and bh[i] > bh[i-1] and bh[i] > bh[i-2] and bh[i] > bh[i+1] and bh[i] > bh[i+2]:
            sw_h.append(bh[i])
        if len(sw_l) < 2 and bl[i] < bl[i-1] and bl[i] < bl[i-2] and bl[i] < bl[i+1] and bl[i] < bl[i+2]:
            sw_l.append(bl[i])
    if len(sw_h) < 2 or len(sw_l) < 2:
        return 0, rng_hi, rng_lo

    if c1 > sw_h[0]:
        b = 1                                   # BOS up
    elif c1 < sw_l[0]:
        b = -1                                  # BOS down
    elif sw_h[0] > sw_h[1] and sw_l[0] > sw_l[1]:
        b = 1                                   # HH + HL
    elif sw_h[0] < sw_h[1] and sw_l[0] < sw_l[1]:
        b = -1                                  # LH + LL
    else:
        b = 0
    return b, rng_hi, rng_lo


def find_setup(H, L, O, C, T, atr, s1, s2, rng_hi, rng_lo, t1, t2, mk):
    """Buy-side setup finder. Sells are found by mirroring prices (x -1).
    0.0 for a level means 'not available'. mk = market settings dict."""
    SW, LB, SB = P["sweep_window"], P["lookback"], P["struct_bars"]
    score, why = 0, []

    # 1) liquidity sweep
    sweep_idx, sweep_low = 1, L[1]
    for i in range(1, SW + 1):
        if L[i] < sweep_low:
            sweep_low, sweep_idx = L[i], i
    swing_low = np.nanmin(L[SW + 1:SW + LB + 1])

    ss = 0
    if s1 != 0.0 and sweep_low < s1 and C[1] > s1:
        ss = 2; why.append("PD-sweep")
    if s2 != 0.0 and sweep_low < s2 and C[1] > s2:
        ss = 2; why.append("Asian-sweep")
    if ss == 0 and sweep_low < swing_low and C[1] > swing_low:
        ss = 1; why.append("Swing-sweep")
    if ss == 0:
        return None
    score += ss

    # 2) MSS / CHoCH
    minor_high = np.nanmax(H[sweep_idx + 1:sweep_idx + SB + 1])
    if not any(C[k] > minor_high for k in range(1, sweep_idx + 1)):
        return None
    score += 2; why.append("MSS")

    # 3) FVG from displacement (middle candle at/after sweep)
    fi, fvg_top, fvg_bot = -1, 0.0, 0.0
    for i in range(1, sweep_idx):
        m = i + 1
        if (C[m] - O[m]) >= P["min_disp_atr"] * atr and L[i] > H[i + 2]:
            fi, fvg_top, fvg_bot = i, L[i], H[i + 2]
            break
    if fi < 0:
        return None
    score += 2; why.append("FVG")

    # 4) Order block (last down candle before displacement) + overlap entry
    entry = (fvg_top + fvg_bot) / 2.0
    m = fi + 1
    for j in range(m + 1, m + 5):
        if C[j] < O[j]:
            ob_hi, ob_lo = H[j], L[j]
            if ob_lo <= fvg_top and ob_hi >= fvg_bot:
                entry = (min(ob_hi, fvg_top) + max(ob_lo, fvg_bot)) / 2.0
                score += 2; why.append("OB+FVG")
            break

    # already mitigated?
    for k in range(1, fi):
        if L[k] <= entry:
            return None

    # 5) premium / discount + OTE
    rng = rng_hi - rng_lo
    if rng <= 0:
        return None
    eq = (rng_hi + rng_lo) / 2.0
    if entry < eq:
        score += 2; why.append("Discount")
    elif P["require_discount"]:
        return None
    retr = (rng_hi - entry) / rng
    if 0.62 <= retr <= 0.79:
        score += 1; why.append("OTE")

    if score < P["min_score"]:
        return None

    sl = sweep_low - P["sl_buf_atr"] * atr
    risk = entry - sl
    if risk <= 0 or risk < P["min_sl_atr"] * atr or risk > P["max_sl_atr"] * atr:
        return None
    if risk * mk["lot"] * mk["contract"] * mk["usd"] > mk["max_loss_usd"]:
        return None

    # TP = nearest liquidity pool giving >= min RR (capped)
    highest = np.nanmax(H[1:61])
    min_tp = entry + P["min_rr"] * risk
    cands = [c for c in (t1, t2, highest) if c != 0.0 and c >= min_tp]
    tp = entry + P["default_rr"] * risk if not cands else min(min(cands), entry + P["max_rr"] * risk)

    return dict(entry=entry, sl=sl, tp=tp, score=score, why=why,
                sweep_time=T[sweep_idx], fvg=(fvg_bot, fvg_top))


# ----------------------------------------------------------------- output
def make_chart(df, sig, title, dec=2):
    d = df.tail(70)
    n = len(d)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    for i, (_, r) in enumerate(d.iterrows()):
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        col = "#26a69a" if c >= o else "#ef5350"
        ax.vlines(i, l, h, color=col, lw=1)
        ax.add_patch(plt.Rectangle((i - 0.35, min(o, c)), 0.7, max(abs(c - o), 1e-6), color=col))

    e, sl, tp = sig["entry"], sig["sl"], sig["tp"]
    x0, x1 = n - 25, n + 8
    if sig.get("fvg"):
        ax.fill_between([x0, x1], sig["fvg"][0], sig["fvg"][1], color="royalblue", alpha=0.18)
    ax.fill_between([x0, x1], e, sl, color="red", alpha=0.13)
    ax.fill_between([x0, x1], e, tp, color="green", alpha=0.13)
    for y, col, lab in ((e, "dodgerblue", "ENTRY"), (sl, "red", "SL"), (tp, "green", "TP")):
        ax.hlines(y, x0, x1, colors=col, lw=1.8)
        ax.text(x1 + 0.4, y, f"{lab} {y:,.{dec}f}", color=col, va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(-1, n + 19)
    pos = list(range(0, n, 10))
    ax.set_xticks(pos)
    ax.set_xticklabels([d.index[i].tz_convert(LONDON).strftime("%H:%M") for i in pos])
    ax.set_xlabel("UK time")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def send_discord(text, img=None):
    if not WEBHOOK:
        print("[no DISCORD_WEBHOOK set]\n" + text)
        return
    if img is not None:
        r = requests.post(WEBHOOK, data={"content": text},
                          files={"file": ("signal.png", img, "image/png")}, timeout=30)
    else:
        r = requests.post(WEBHOOK, json={"content": text}, timeout=30)
    print("Discord status:", r.status_code)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


# ----------------------------------------------------------------- main
def run_market(name, now_utc, now_ldn, st):
    mk = MARKETS[name]
    tk = mk["ticker"]

    if now_utc.weekday() >= 5 and not mk["weekend"]:
        print(name, "weekend band"); return
    if not zone_ok(mk, now_utc, now_ldn):
        print(name, "trading window nahi", now_ldn); return

    m15 = fetch(tk, "15m", f"{mk['days']}d", 15)
    h1 = fetch(tk, "60m", "30d", 60)
    d1 = fetch(tk, "1d", "15d", 0, drop_forming=False)
    if m15 is None or h1 is None or d1 is None or len(m15) < 150:
        print(name, "data kam hai"); return

    bias, rng_hi, rng_lo = bias_and_range(h1)
    if bias == 0:
        print(name, "bias neutral"); return

    hi, lo, op, cl = (series(m15[c]) for c in ("high", "low", "open", "close"))
    T = [None] + list(m15.index[::-1])
    tr = pd.concat([m15["high"] - m15["low"],
                    (m15["high"] - m15["close"].shift()).abs(),
                    (m15["low"] - m15["close"].shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(P["atr_period"]).mean().iloc[-1])
    if not atr or np.isnan(atr):
        return

    prev = d1[d1.index.date < now_utc.date()]
    pdh = float(prev["high"].iloc[-1]) if len(prev) else 0.0
    pdl = float(prev["low"].iloc[-1]) if len(prev) else 0.0

    as_h = as_l = 0.0
    today = m15[(m15.index.date == now_utc.date()) & (m15.index.hour < P["asian_end"])]
    if len(today):
        as_h, as_l = float(today["high"].max()), float(today["low"].min())

    if bias == 1:
        sig = find_setup(hi, lo, op, cl, T, atr, pdl, as_l, rng_hi, rng_lo, pdh, as_h, mk)
    else:
        sig = find_setup(-lo, -hi, -op, -cl, T, atr, -pdh, -as_h, -rng_lo, -rng_hi, -pdl, -as_l, mk)
        if sig:
            sig["entry"], sig["sl"], sig["tp"] = -sig["entry"], -sig["sl"], -sig["tp"]
            sig["fvg"] = (-sig["fvg"][1], -sig["fvg"][0])
    if not sig:
        print(name, "setup nahi"); return

    price = float(m15["close"].iloc[-1])
    if (bias == 1 and sig["entry"] >= price) or (bias == -1 and sig["entry"] <= price):
        print(name, "price entry se guzar chuka"); return

    key = str(sig["sweep_time"])
    if st.get(name) == key:
        print(name, "ye setup pehle bhej diya"); return

    e, sl, tp = sig["entry"], sig["sl"], sig["tp"]
    rr = abs(tp - e) / abs(e - sl)
    exp = (now_ldn + pd.Timedelta(minutes=15 * P["expiry_bars"])).strftime("%H:%M")
    side = "🟢 BUY" if bias == 1 else "🔴 SELL"
    d = mk["dec"]
    k = mk["lot"] * mk["contract"] * mk["usd"]          # price distance -> USD
    loss_usd, win_usd = abs(e - sl) * k, abs(tp - e) * k
    pip = PIP.get(name)
    if pip:
        pl = f"SL {abs(e - sl) / pip:,.1f} pips | TP {abs(tp - e) / pip:,.1f} pips\n"
    else:
        pl = ""
    msg = (f"{side} {name} LIMIT | Score {sig['score']}/11\n"
           f"Entry `{e:,.{d}f}` | SL `{sl:,.{d}f}` | TP `{tp:,.{d}f}` | RR 1:{rr:.1f}\n"
           f"{pl}"
           f"💰 {mk['lot']} lot par: Loss ~${loss_usd:,.2f} | Profit ~${win_usd:,.2f}\n"
           f"Lot {mk['lot']} | {exp} UK tak valid\n"
           f"Feed price abhi: `{price:,.{d}f}` (MT5 price se farq nikalo)\n"
           f"{' + '.join(sig['why'])}\n"
           f"⚠️ {mk['note']}")
    title = f"{name} M15 {'BUY' if bias == 1 else 'SELL'} | Score {sig['score']}"
    send_discord(msg, make_chart(m15, sig, title, d))
    st[name] = key
    save_state(st)


def run_once():
    now_utc = pd.Timestamp.now(tz="UTC")
    now_ldn = now_utc.tz_convert(LONDON)

    if os.getenv("TEST") == "1":
        # Test sends only a few markets (to avoid spamming Discord): Gold, EURUSD, BTC.
        # Set TEST_MARKETS (comma list) to test others.
        test_list = [m.strip().upper() for m in os.getenv("TEST_MARKETS", "XAUUSD,EURUSD,BTCUSD,NAS100").split(",")]
        for name in [m for m in test_list if m in MARKETS]:
            mk = MARKETS[name]
            df = fetch(mk["ticker"], "15m", f"{mk['days']}d", 15)
            if df is None:
                send_discord(f"Test {name}: data nahi mili (yfinance)."); continue
            d = mk["dec"]
            px = float(df["close"].iloc[-1])
            step = float(df["close"].diff().abs().tail(50).mean()) * 8
            sig = dict(entry=px, sl=px - step, tp=px + 2.5 * step, fvg=(px - step * 0.4, px - step * 0.1))
            send_discord(f"✅ Test OK - {name} par bot Discord se connected hai. (Ye sirf demo chart hai, signal nahi.)\n"
                         f"Feed ka last price: `{px:,.{d}f}` | candles: {len(df)} | aakhri candle: {df.index[-1].tz_convert(LONDON):%a %H:%M} UK\n"
                         f"Is price ko MT5 ke price se milao.",
                         make_chart(df, sig, f"TEST | {name} M15", d))
            time.sleep(1)
        return

    st = load_state()
    for name in ACTIVE:
        try:
            run_market(name, now_utc, now_ldn, st)
        except Exception as ex:          # one market failing must not stop the others
            print(name, "error:", ex)
        time.sleep(1.5)                  # be gentle with the free data source


if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True:
            try:
                run_once()
            except Exception as ex:
                print("Error:", ex)
            time.sleep(120)
    else:
        run_once()
