"""
Daily NSE breakout scanner (end-of-day, technical rules).
Run after market close. Sends results to Telegram and saves a CSV.

Rules (all must be true on the latest daily candle):
  1. Close > highest HIGH of the previous N days (default 55)   -> price breakout
  2. Volume >= 1.5 x 20-day average volume                      -> volume confirmation
  3. Close in the top 30% of the day's range                    -> closed near the high
  4. Close > 50-DMA and 50-DMA > 200-DMA                        -> uptrend
  5. RSI(14) between 55 and 80                                  -> momentum, not overheated
  6. Close >= Rs 50 and 20-day avg traded value >= Rs 5 crore   -> liquidity filter
"""
import io
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- settings (change freely) ----------------
BREAKOUT_LOOKBACK = 55      # days; 252 = strict 52-week high breakout
VOL_MULTIPLE = 1.5
CLOSE_POSITION_MIN = 0.70   # close must be in top 30% of day's range
RSI_MIN, RSI_MAX = 55, 80
MIN_PRICE = 50
MIN_AVG_VALUE_CR = 5        # 20-day avg traded value, Rs crore
ATR_STOP_MULT = 1.5
TOP_N_IN_MESSAGE = 15
CHUNK = 100

NSE_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
IST = timezone(timedelta(hours=5, minutes=30))


def load_symbols():
    """Nifty 500 list from NSE; falls back to symbols.txt if NSE blocks the request."""
    try:
        r = requests.get(NSE_LIST_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        syms = df["Symbol"].dropna().str.strip().tolist()
        print(f"Loaded {len(syms)} symbols from NSE")
        return syms
    except Exception as e:
        print(f"NSE list download failed ({e}); using symbols.txt")
        with open(os.path.join(os.path.dirname(__file__), "symbols.txt")) as f:
            return [s.strip() for s in f if s.strip() and not s.startswith("#")]


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def evaluate(sym, df):
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 210:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    last = df.iloc[-1]
    prev_high = h.shift(1).rolling(BREAKOUT_LOOKBACK).max().iloc[-1]
    prev_high_52w = h.shift(1).rolling(252).max().iloc[-1]
    vol_avg = v.shift(1).rolling(20).mean().iloc[-1]
    value_cr = (c * v).rolling(20).mean().iloc[-1] / 1e7
    rng = last["High"] - last["Low"]
    close_pos = (last["Close"] - last["Low"]) / rng if rng > 0 else 0
    dma50, dma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    r = rsi(c).iloc[-1]
    a = atr(df).iloc[-1]
    vol_ratio = last["Volume"] / vol_avg if vol_avg else 0

    ok = (
        last["Close"] > prev_high
        and vol_ratio >= VOL_MULTIPLE
        and close_pos >= CLOSE_POSITION_MIN
        and last["Close"] > dma50 > dma200
        and RSI_MIN <= r <= RSI_MAX
        and last["Close"] >= MIN_PRICE
        and value_cr >= MIN_AVG_VALUE_CR
    )
    if not ok:
        return None
    return {
        "Symbol": sym,
        "Close": round(last["Close"], 2),
        "Breakout_Level": round(prev_high, 2),
        "Pct_Above_Level": round((last["Close"] / prev_high - 1) * 100, 2),
        "Vol_x_Avg": round(vol_ratio, 2),
        "RSI": round(r, 1),
        "52W_High_Breakout": bool(last["Close"] > prev_high_52w),
        "Stop_Loss_ATR": round(last["Close"] - ATR_STOP_MULT * a, 2),
        "Risk_%": round(ATR_STOP_MULT * a / last["Close"] * 100, 2),
    }


def fetch_all(symbols):
    tickers = [f"{s}.NS" for s in symbols]
    frames = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        try:
            data = yf.download(chunk, period="2y", interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print(f"chunk {i} failed: {e}")
            continue
        for t in chunk:
            try:
                d = data[t].dropna(how="all")
                if not d.empty:
                    frames[t[:-3]] = d
            except Exception:
                pass
    return frames


def send_telegram(text):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram secrets not set; skipping send.")
        return
    for part in [text[i : i + 4000] for i in range(0, len(text), 4000)]:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": part}, timeout=30)


def main():
    today = datetime.now(IST).date()
    frames = fetch_all(load_symbols())
    if not frames:
        send_telegram("Breakout scan failed: no price data downloaded.")
        sys.exit(1)

    last_dates = pd.Series([d.index[-1].date() for d in frames.values()])
    session = last_dates.mode().iloc[0]
    if session != today and os.getenv("FORCE_RUN") != "1":
        msg = f"No new trading session today ({today}). Latest data is {session}. Market holiday or data not updated."
        print(msg)
        send_telegram(msg)
        return

    rows = [r for s, d in frames.items() if (r := evaluate(s, d))]
    out = pd.DataFrame(rows)
    if out.empty:
        send_telegram(f"Breakout scan {session}: no stocks met all criteria.")
        print("No breakouts.")
        return

    out = out.sort_values(["52W_High_Breakout", "Vol_x_Avg"], ascending=False).reset_index(drop=True)
    out.to_csv(f"breakouts_{session}.csv", index=False)
    print(out.to_string())

    lines = [f"NSE BREAKOUTS - {session} ({len(out)} found, Nifty 500 scan)", ""]
    for _, r in out.head(TOP_N_IN_MESSAGE).iterrows():
        tag = " [52W HIGH]" if r["52W_High_Breakout"] else ""
        lines.append(f"{r.Symbol}{tag}\n  Close {r.Close} | Broke {r.Breakout_Level} (+{r.Pct_Above_Level}%)"
                     f"\n  Vol {r.Vol_x_Avg}x | RSI {r.RSI} | ATR stop {r.Stop_Loss_ATR} (risk {r['Risk_%']}%)")
    lines += ["", "Rule-based screen, not advice. Verify with a SEBI-registered analyst and check the chart."]
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
