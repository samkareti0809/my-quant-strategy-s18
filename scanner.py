import io
import os
import smtplib
import requests
import warnings
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings('ignore')

# =====================================================================
# 1. DATABASE MANAGER (SQLITE: DAILY SCANS & TRADE JOURNAL)
# =====================================================================
class QuantDatabaseManager:
    def __init__(self, db_name="strategy_journal.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        # Table 1: Archive of all daily qualified signals
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT,
            ticker TEXT,
            signal TEXT,
            price REAL,
            stop_loss REAL,
            risk_pct REAL,
            target TEXT,
            notes TEXT
        )
        """)
        # Table 2: Trade lifecycle tracking (profit/loss booking)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            entry_date TEXT,
            entry_price REAL,
            stop_loss REAL,
            status TEXT,
            exit_date TEXT,
            exit_price REAL,
            pnl_pct REAL,
            exit_reason TEXT
        )
        """)
        self.conn.commit()

    def log_daily_scan(self, triggered_signals: list):
        scan_date = datetime.now().strftime("%Y-%m-%d")
        cursor = self.conn.cursor()
        for sig in triggered_signals:
            cursor.execute("""
                INSERT INTO daily_scans (scan_date, ticker, signal, price, stop_loss, risk_pct, target, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_date, sig['Ticker'], sig['Signal'], sig['Price'], 
                sig['Stop_Loss'], sig['Risk_%'], str(sig['Target']), sig['Notes']
            ))
        self.conn.commit()
        print(f"[*] Archived {len(triggered_signals)} scanned signals to SQLite database.")

# =====================================================================
# 2. RUNTIME IN-MEMORY NSE TICKER INGESTION & SCANNER
# =====================================================================
def get_live_nse_universe(series_filter: str = "EQ") -> list:
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, timeout=10)
    response = session.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch NSE equity list. HTTP Status: {response.status_code}")
        
    df = pd.read_csv(io.StringIO(response.text))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    
    if series_filter:
        df = df[df['SERIES'] == series_filter]
        
    raw_symbols = df['SYMBOL'].dropna().unique().tolist()
    return [f"{sym}.NS" for sym in raw_symbols]

def analyze_ticker(symbol: str) -> dict:
    try:
        df = yf.download(symbol, period="120d", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 50:
            return None

        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(14).mean()
        
        plus_dm = df['High'].diff().where(lambda x: (x > -df['Low'].diff()) & (x > 0), 0.0)
        minus_dm = (-df['Low'].diff()).where(lambda x: (x > df['High'].diff()) & (x > 0), 0.0)
        tr_sum = df['TR'].rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / (tr_sum + 1e-9))
        minus_di = 100 * (minus_dm.rolling(14).sum() / (tr_sum + 1e-9))
        df['ADX'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)).rolling(14).mean()
        
        df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
        df['Upper_Climax'] = df['EMA20'] + (3.0 * df['ATR'])

        c, o, h, l = df['Close'].iloc[-1], df['Open'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1]
        prev_c = df['Close'].iloc[-2]
        e10, e20, e50, e200 = df['EMA10'].iloc[-1], df['EMA20'].iloc[-1], df['EMA50'].iloc[-1], df['EMA200'].iloc[-1]
        prev_e10 = df['EMA10'].iloc[-2]
        atr, adx, vol, vol_avg = df['ATR'].iloc[-1], df['ADX'].iloc[-1], df['Volume'].iloc[-1], df['Vol_SMA20'].iloc[-1]
        vol_ratio = round(vol / (vol_avg + 1e-9), 2)

        macro_bull = (c > e200) and (e50 > e200)
        slope_50 = e50 > df['EMA50'].iloc[-10]

        breakout = (prev_c <= prev_e10) and (c > e10) and (c > o)
        bounce = (l <= e10 * 1.01) and (c > e10) and (c > o) and (e10 > e20)
        is_tag_b = (breakout or bounce) and (c > e20) and (vol >= 0.70 * vol_avg) and (adx >= 18)

        if is_tag_b and macro_bull and slope_50:
            raw_stop = min(df['Low'].iloc[-3:].min(), e20 - (1.2 * atr))
            stop = max(c * 0.92, raw_stop)
            risk_pct = round(((c - stop) / c) * 100, 2)
            
            stop_type = "Hard -8% Cap" if risk_pct >= 8.0 else "Technical Swing Low"
            tier = "Institutional Mid/High Cap" if c > 500 else "Small Cap Tier"
            notes = f"Pattern Breakout | Vol: {vol_ratio}x SMA20 | Stop: {stop_type} ({risk_pct}%) | Tier: {tier}"

            return {
                "Ticker": symbol, "Signal": "BUY_TREND_MARKUP", "Price": round(c, 2),
                "Stop_Loss": round(stop, 2), "Risk_%": risk_pct, "Target": "Trailing 20 EMA / Climax Flag", "Notes": notes
            }

        if (h >= df['Upper_Climax'].iloc[-1]) and (c > e10):
            notes = "Climax Exhaustion Flag (🚩) | Stretched >= 3.0 ATR above 20 EMA"
            return {
                "Ticker": symbol, "Signal": "SELL_CLIMAX_FLAG", "Price": round(c, 2),
                "Stop_Loss": round(e10, 2), "Risk_%": 0.0, "Target": "Immediate Scale Out", "Notes": notes
            }
    except Exception:
        return None
    return None

def run_live_pipeline() -> list:
    tickers = get_live_nse_universe(series_filter="EQ")
    print(f"[*] Scanning {len(tickers)} stocks for pattern triggers...")
    triggered = []
    for symbol in tickers:
        res = analyze_ticker(symbol)
        if res:
            triggered.append(res)
    return triggered

# =====================================================================
# 3. PRIORITY SCORING & TOP 3 SELECTION (APPLYING 4-FACTOR MATRIX)
# =====================================================================
def select_top_3_picks(triggered_signals: list) -> list:
    buy_signals = [s for s in triggered_signals if s['Signal'] == 'BUY_TREND_MARKUP']
    if not buy_signals:
        return []

    scored = []
    for sig in buy_signals:
        notes = sig['Notes']
        vol_ratio = 1.0
        if "Vol: " in notes:
            try:
                vol_ratio = float(notes.split("Vol: ")[1].split("x")[0])
            except ValueError:
                pass
        
        # Priority Matrix Scoring: Rewards high volume, institutional price tier (> ₹500), tight risk (< 6.5%)[cite: 2]
        score = (vol_ratio * 4.0) - (sig['Risk_%'] * 0.8)
        if sig['Price'] >= 500:
            score += 6.0
        scored.append((score, sig))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:3]]

# =====================================================================
# 4. NOTIFICATION DISPATCHERS (TELEGRAM & EMAIL)
# =====================================================================
def send_telegram_alert(top_signals: list):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    msg = "🚀 *Elite Tier 1 Quant Watchlist (Top 3)*\n\n"
    for idx, t in enumerate(top_signals, 1):
        msg += f"*Rank #{idx}: {t['Ticker']}*\n"
        msg += f"• *Price:* ₹{t['Price']} | *Risk:* {t['Risk_%']}%\n"
        msg += f"• *Stop Loss:* ₹{t['Stop_Loss']}\n"
        msg += f"• *Highlights:* {t['Notes']}\n\n"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=10)

def send_email_report(top_signals: list):
    sender_email, sender_password, receiver_email = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD"), os.environ.get("RECEIVER_EMAIL")
    if not sender_email or not sender_password or not receiver_email:
        return

    html = "<h2>🚀 Elite Tier 1 Strategy Report: Top 3 Picks</h2><table><tr><th>Rank</th><th>Ticker</th><th>Price</th><th>Stop Loss</th><th>Risk %</th><th>Rationale</th></tr>"
    for idx, t in enumerate(top_signals, 1):
        html += f"<tr><td>#{idx}</td><td><b>{t['Ticker']}</b></td><td>₹{t['Price']}</td><td>₹{t['Stop_Loss']}</td><td>{t['Risk_%']}%</td><td>{t['Notes']}</td></tr>"
    html += "</table>"

    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = "📈 Quant Strategy Report: Top 3 Picks", sender_email, receiver_email
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

# =====================================================================
# 5. EXECUTION CONTROLLER
# =====================================================================
if __name__ == "__main__":
    db = QuantDatabaseManager("strategy_journal.db")
    signals = run_live_pipeline()
    
    if signals:
        db.log_daily_scan(signals)
        top_3 = select_top_3_picks(signals)
        if top_3:
            send_telegram_alert(top_3)
            send_email_report(top_3)
            print("[*] Successfully completed scan, database logging, and dispatching alerts.")
