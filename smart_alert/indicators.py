"""Port ของ Pine Script "Smart signal V2.3" เป็น Python (numpy)
คำนวณทีละแท่งให้ตรงกับพฤติกรรมของ Pine ให้มากที่สุด"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NAN = np.nan


# ---------- ta.* พื้นฐาน ----------
def sma(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), NAN)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).mean(axis=1)
    return out


def stdev(x, n):
    """Pine ta.stdev = population stdev (ddof=0)"""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), NAN)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).std(axis=1)
    return out


def rma(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), NAN)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def ema(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), NAN)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def atr(high, low, close, n):
    h, l, c = (np.asarray(v, dtype=float) for v in (high, low, close))
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    prev = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - prev), np.abs(l[1:] - prev)])
    return rma(tr, n)


# ---------- Follow Line ----------
def follow_line(close, high, low, upper, lower, a):
    """คืน (trendline, itrend) โดย itrend: 1 / -1 / 0(ยังไม่รู้)"""
    n = len(close)
    tl = np.full(n, NAN)
    it = np.zeros(n, dtype=int)
    for i in range(n):
        prev = tl[i - 1] if i > 0 else NAN
        if close[i] > upper[i]:          # nan -> False เหมือน Pine
            v = low[i] - a[i]
            if v < prev:
                v = prev
        elif close[i] < lower[i]:
            v = high[i] + a[i]
            if v > prev:
                v = prev
        else:
            v = prev
        tl[i] = v
        it[i] = it[i - 1] if i > 0 else 0
        if v > prev:
            it[i] = 1
        elif v < prev:
            it[i] = -1
    return tl, it


def follow_line_bb(o, h, l, c, period=21, dev=1.0, use_atr=True, atr_period=5):
    """Trend Line ชุดที่ 1 (Bollinger + ATR)"""
    basis = sma(c, period)
    sd = stdev(c, period)
    a = atr(h, l, c, atr_period) if use_atr else np.zeros(len(c))
    return follow_line(c, h, l, basis + sd * dev, basis - sd * dev, a)


def follow_line_ma(o, h, l, c, ma_type="EMA", length=21, dev=1.0, atr_period=5):
    """Trend Line ชุดที่ 2 (Smart Follow Line) รองรับ EMA/SMA/RMA"""
    if ma_type == "EMA":
        basis = ema(c, length)
    elif ma_type == "SMA":
        basis = sma(c, length)
    elif ma_type == "RMA":
        basis = rma(c, length)
    else:
        raise ValueError(f"MA type {ma_type} ยังไม่ได้ port (รองรับ EMA/SMA/RMA)")
    sd = stdev(c, length)
    a = atr(h, l, c, atr_period)
    return follow_line(c, h, l, basis + sd * dev, basis - sd * dev, a)


# ---------- Kernel Regression (Rational Quadratic) ----------
def kernel_rq(src, lookback=8, rel_weight=1.0, start_at=20, filt="Smooth"):
    src = np.asarray(src, dtype=float)
    n = len(src)
    m = 1 + start_at + 1  # i = 0 .. size+startAtBar
    i = np.arange(m)
    w = (1 + (i ** 2) / (lookback ** 2 * 2 * rel_weight)) ** (-rel_weight)
    wsum = w.sum()

    def smooth(x):
        out = np.full(len(x), NAN)
        valid = ~np.isnan(x)
        # ต้องมีข้อมูลครบ m แท่งติดกัน
        for t in range(m - 1, len(x)):
            win = x[t - m + 1:t + 1][::-1]
            if np.isnan(win).any():
                continue
            out[t] = (win * w).sum() / wsum
        return out

    y1 = smooth(src)
    y2 = smooth(y1)
    if filt == "No Filter":
        return y1
    if filt == "Zero Lag":
        return y1 + (y1 - y2)
    return y2  # Smooth


def kernel_dir(k):
    d = np.zeros(len(k), dtype=int)
    for i in range(1, len(k)):
        if not (np.isnan(k[i]) or np.isnan(k[i - 1])):
            d[i] = 1 if k[i] > k[i - 1] else -1
    return d


# ---------- ZigZag swing (lastSwingHigh / lastSwingLow) ----------
def zigzag_swings(high, low, length=13):
    n = len(high)
    last_hi = np.full(n, NAN)
    last_lo = np.full(n, NAN)
    cur_hi, cur_lo = NAN, NAN
    d = 0
    zz = []  # index 0 = ล่าสุด; แต่ละตัว [y, 'H'|'L']
    for i in range(n):
        kh = kl = NAN
        if i >= length - 1:
            hw = high[i - length + 1:i + 1]
            lw = low[i - length + 1:i + 1]
            if high[i] >= hw.max():
                kh = high[i]
            if low[i] <= lw.min():
                kl = low[i]
        prev_dir = d
        if not np.isnan(kh) and np.isnan(kl):
            d = 1
        elif not np.isnan(kl) and np.isnan(kh):
            d = -1
        if not (np.isnan(kh) and np.isnan(kl)):
            y = kh if d == 1 else kl
            s = "H" if d == 1 else "L"
            if d != prev_dir:
                zz.insert(0, [y, s])
                if len(zz) > 4:
                    zz.pop()
            elif not zz:
                zz.insert(0, [y, s])
            else:
                if (d == 1 and y > zz[0][0]) or (d == -1 and y < zz[0][0]):
                    zz[0] = [y, s]
        if len(zz) >= 2:
            if zz[0][1] == "H":
                cur_hi = zz[0][0]
            if zz[0][1] == "L":
                cur_lo = zz[0][0]
        last_hi[i] = cur_hi
        last_lo[i] = cur_lo
    return last_hi, last_lo
