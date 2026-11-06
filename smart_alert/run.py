"""ดึงราคา -> คำนวณ Confluence (Status 3 ตัว + Smart Buy/Sell) -> ส่ง Telegram -> เขียน docs/data.json

ใช้งาน:  python run.py            (ดึงราคาจริงจาก Binance)
         python run.py --demo     (ใช้ข้อมูลจำลอง ไม่ต้องต่อเน็ต)
ตัวแปรสภาพแวดล้อม: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

import indicators as ind

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
OUT = ROOT.parent / "docs" / "data.json"
STATE = ROOT / "state.json"

TF_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
          "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}


# ---------- ดึงข้อมูล ----------
def fetch_klines(symbol, interval, limit=1000):
    r = requests.get(f"{CFG['api_base']}/api/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=20)
    r.raise_for_status()
    rows = r.json()
    return {
        "t": np.array([x[0] // 1000 for x in rows], dtype=np.int64),
        "o": np.array([float(x[1]) for x in rows]),
        "h": np.array([float(x[2]) for x in rows]),
        "l": np.array([float(x[3]) for x in rows]),
        "c": np.array([float(x[4]) for x in rows]),
    }


CAP_RES = {"1m": "MINUTE", "5m": "MINUTE_5", "15m": "MINUTE_15", "30m": "MINUTE_30",
           "1h": "HOUR", "4h": "HOUR_4", "1d": "DAY"}


def capital_login():
    """เปิด session ของ Capital.com (ต้องมี CAPITAL_API_KEY / CAPITAL_IDENTIFIER / CAPITAL_PASSWORD)"""
    key, ident, pw = (os.getenv("CAPITAL_API_KEY"), os.getenv("CAPITAL_IDENTIFIER"),
                      os.getenv("CAPITAL_PASSWORD"))
    if not (key and ident and pw):
        raise SystemExit("ไม่พบ CAPITAL_API_KEY / CAPITAL_IDENTIFIER / CAPITAL_PASSWORD ใน Secrets")
    base = CFG.get("capital_base", "https://demo-api-capital.backend-capital.com")
    r = requests.post(f"{base}/api/v1/session",
                      headers={"X-CAP-API-KEY": key, "Content-Type": "application/json"},
                      json={"identifier": ident, "password": pw}, timeout=20)
    if not r.ok:
        raise SystemExit(f"Capital login ล้มเหลว {r.status_code}: {r.text[:200]}")
    return base, {"CST": r.headers["CST"], "X-SECURITY-TOKEN": r.headers["X-SECURITY-TOKEN"]}


def fetch_capital(epic, interval, auth, limit=1000):
    base, hdr = auth
    r = requests.get(f"{base}/api/v1/prices/{epic}", headers=hdr, timeout=20,
                     params={"resolution": CAP_RES[interval], "max": limit})
    r.raise_for_status()
    rows = r.json()["prices"]
    from datetime import datetime, timezone

    def ts(x):
        return int(datetime.fromisoformat(x["snapshotTimeUTC"]).replace(tzinfo=timezone.utc).timestamp())

    def mid(x, f):  # ใช้ราคากลาง bid/ask
        return (float(x[f]["bid"]) + float(x[f]["ask"])) / 2

    rows.sort(key=ts)
    return {
        "t": np.array([ts(x) for x in rows], dtype=np.int64),
        "o": np.array([mid(x, "openPrice") for x in rows]),
        "h": np.array([mid(x, "highPrice") for x in rows]),
        "l": np.array([mid(x, "lowPrice") for x in rows]),
        "c": np.array([mid(x, "closePrice") for x in rows]),
    }


def demo_klines(interval, n=1000, seed=7):
    rng = np.random.default_rng(seed + TF_SEC[interval])
    step = TF_SEC[interval]
    end = int(time.time()) // step * step
    t = np.arange(end - (n - 1) * step, end + 1, step, dtype=np.int64)
    ret = rng.normal(0, 0.0012, n) + 0.0004 * np.sin(np.arange(n) / 60)
    c = 2600 * np.exp(np.cumsum(ret))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0006, n)))
    return {"t": t, "o": o, "h": h, "l": l, "c": c}


def trim_closed(k, tf):
    """ตัดแท่งที่ยังไม่ปิดออก (เหมือน alert once per bar close)"""
    if k["t"][-1] + TF_SEC[tf] > time.time():
        return {key: v[:-1] for key, v in k.items()}
    return k


# ---------- คำนวณ ----------
def status_1h(k):
    o, h, l, c = k["o"], k["h"], k["l"], k["c"]
    _, t1 = ind.follow_line_bb(o, h, l, c, CFG["bb_period"], CFG["bb_dev"],
                               CFG["bb_use_atr"], CFG["bb_atr_period"])
    _, t2 = ind.follow_line_ma(o, h, l, c, CFG["ma_type"], CFG["ma_length"],
                               CFG["ma_dev"], CFG["ma_atr_period"])
    kn = ind.kernel_rq(c, CFG["kernel_lookback"], CFG["kernel_weight"],
                       CFG["kernel_level"], CFG["kernel_filter"])
    return t1, t2, ind.kernel_dir(kn)


def compute(k, kh):
    o, h, l, c, t = k["o"], k["h"], k["l"], k["c"], k["t"]
    n = len(c)

    # Smart Buy/Sell (Follow Line MA บน TF ของกราฟ)
    tl2, it2 = ind.follow_line_ma(o, h, l, c, CFG["ma_type"], CFG["ma_length"],
                                  CFG["ma_dev"], CFG["ma_atr_period"])
    buy = np.zeros(n, bool)
    sell = np.zeros(n, bool)
    buy[1:] = (it2[:-1] == -1) & (it2[1:] == 1)
    sell[1:] = (it2[:-1] == 1) & (it2[1:] == -1)

    last_hi, last_lo = ind.zigzag_swings(h, l, CFG["zigzag_length"])

    # Status 1H -> map เข้าแท่งของกราฟ
    s1, s2, s3 = status_1h(kh)
    st_sec = TF_SEC[CFG["status_timeframe"]]
    ht = kh["t"]
    idx = np.searchsorted(ht, (t // st_sec) * st_sec, side="right") - 1
    idx = np.clip(idx, 0, len(ht) - 1)
    m1, m2, m3 = s1[idx], s2[idx], s3[idx]
    all_bull = (m1 == 1) & (m2 == 1) & (m3 == 1)
    all_bear = (m1 == -1) & (m2 == -1) & (m3 == -1)

    conf_buy = buy & ~np.isnan(last_lo) & all_bull
    conf_sell = sell & ~np.isnan(last_hi) & all_bear
    return dict(tl2=tl2, it2=it2, buy=buy, sell=sell, last_hi=last_hi, last_lo=last_lo,
                m1=m1, m2=m2, m3=m3, conf_buy=conf_buy, conf_sell=conf_sell,
                st_last=(int(s1[-1]), int(s2[-1]), int(s3[-1])))


def trade_setup(k, r, i, side):
    entry = float(k["c"][i])
    if side == "buy":
        sl = float(r["last_lo"][i]) - CFG["sl_buffer"]
    else:
        sl = float(r["last_hi"][i]) + CFG["sl_buffer"]
    risk_amt = CFG["balance"] * CFG["risk_percent"] / 100
    dist = abs(entry - sl)
    tp = entry + dist * CFG["rr"] if side == "buy" else entry - dist * CFG["rr"]
    lots = round(abs(risk_amt / (dist * CFG["units_per_lot"])), 2) if dist > 0 else None
    return {"side": side, "entry": round(entry, 4), "sl": round(sl, 4),
            "tp": round(tp, 4), "lots": lots, "risk": round(risk_amt, 2)}


# ---------- แจ้งเตือน ----------
def send_telegram(msg):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[ไม่มี TELEGRAM_TOKEN/TELEGRAM_CHAT_ID] ข้อความที่จะส่ง:\n" + msg)
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": msg}, timeout=15)
    print("telegram:", r.status_code)
    return r.ok


def fmt_alert(tr, bar_time):
    icon = "🟢 BUY" if tr["side"] == "buy" else "🔴 SELL"
    when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(bar_time + CFG["tz_offset_hours"] * 3600))
    return (f"{icon} | {CFG['symbol']} {CFG['timeframe']}\n"
            f"Status 3 ตัว + Smart {'Buy' if tr['side']=='buy' else 'Sell'} ตรงกัน\n"
            f"Entry: {tr['entry']}\nSL: {tr['sl']}\nTP: {tr['tp']}\n"
            f"Lots: {tr['lots']} (risk {CFG['risk_percent']}% = {tr['risk']})\n"
            f"แท่งเวลา {when}")


# ---------- main ----------
def main():
    demo = "--demo" in sys.argv
    tf, stf, sym = CFG["timeframe"], CFG["status_timeframe"], CFG["symbol"]

    source = CFG.get("source", "binance")
    if demo:
        k_all, kh = demo_klines(tf), demo_klines(stf)
    elif source == "capital":
        auth = capital_login()
        sym = CFG["capital_epic"]
        CFG["symbol"] = sym
        k_all, kh = fetch_capital(sym, tf, auth), fetch_capital(sym, stf, auth)
    else:
        k_all, kh = fetch_klines(sym, tf), fetch_klines(sym, stf)

    k = trim_closed(k_all, tf)
    r = compute(k, kh)
    n = len(k["c"])

    # ---- alert (เฉพาะแท่งปิดล่าสุดไม่กี่แท่ง และยังไม่เคยส่ง) ----
    state = json.loads(STATE.read_text()) if STATE.exists() else {"last_alert_time": 0}
    sent = []
    for i in range(max(0, n - CFG["recent_bars_to_alert"]), n):
        side = "buy" if r["conf_buy"][i] else "sell" if r["conf_sell"][i] else None
        if side and int(k["t"][i]) > state["last_alert_time"]:
            tr = trade_setup(k, r, i, side)
            send_telegram(fmt_alert(tr, int(k["t"][i])))
            state["last_alert_time"] = int(k["t"][i])
            sent.append(tr)
    if not demo:
        STATE.write_text(json.dumps(state))

    # ---- data.json สำหรับหน้าเว็บ ----
    off = CFG["tz_offset_hours"] * 3600
    lo = max(0, n - CFG["chart_bars"])
    candles = [{"time": int(k["t"][i]) + off, "open": round(float(k["o"][i]), 4),
                "high": round(float(k["h"][i]), 4), "low": round(float(k["l"][i]), 4),
                "close": round(float(k["c"][i]), 4)} for i in range(lo, n)]

    trend = []
    for i in range(lo, n):
        v = r["tl2"][i]
        if not np.isnan(v):
            trend.append({"time": int(k["t"][i]) + off, "value": round(float(v), 4),
                          "color": "#01f82b" if r["it2"][i] > 0 else "#f44336"})

    markers = []
    for i in range(lo, n):
        tt = int(k["t"][i]) + off
        if r["conf_buy"][i]:
            markers.append({"time": tt, "position": "belowBar", "color": "#00e676",
                            "shape": "arrowUp", "text": "ALL BUY"})
        elif r["buy"][i]:
            markers.append({"time": tt, "position": "belowBar", "color": "#26a69a",
                            "shape": "circle", "text": "Smart Buy"})
        if r["conf_sell"][i]:
            markers.append({"time": tt, "position": "aboveBar", "color": "#ff1744",
                            "shape": "arrowDown", "text": "ALL SELL"})
        elif r["sell"][i]:
            markers.append({"time": tt, "position": "aboveBar", "color": "#f60808",
                            "shape": "circle", "text": "Smart Sell"})

    last_trade = None
    for i in range(n - 1, -1, -1):
        if r["conf_buy"][i] or r["conf_sell"][i]:
            last_trade = trade_setup(k, r, i, "buy" if r["conf_buy"][i] else "sell")
            last_trade["time"] = int(k["t"][i]) + off
            break

    def label(d):
        return "BULL" if d == 1 else "BEAR" if d == -1 else "-"

    data = {
        "symbol": sym, "source": "binance" if (demo or source == "binance") else source,
        "capital_base": CFG.get("capital_base"), "timeframe": tf, "status_timeframe": stf,
        "updated": int(time.time()) + off, "tz_offset": CFG["tz_offset_hours"],
        "status": {"Trend Line (BB)": label(r["st_last"][0]),
                   "Trend Line (MA)": label(r["st_last"][1]),
                   "Line (Kernel)": label(r["st_last"][2])},
        "last_trade": last_trade, "candles": candles, "trend": trend, "markers": markers,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"ok: {n} แท่ง, ALL BUY={int(r['conf_buy'].sum())}, ALL SELL={int(r['conf_sell'].sum())}, "
          f"Smart Buy={int(r['buy'].sum())}, Smart Sell={int(r['sell'].sum())}, ส่ง alert {len(sent)} ครั้ง")


if __name__ == "__main__":
    main()
