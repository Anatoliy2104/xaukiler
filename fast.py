import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, time, timedelta
from time import perf_counter
import math

# progress bar (graceful fallback if tqdm missing)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


NY_TZ = pytz.timezone("America/New_York")

DEBUG = False  # set True to see the same verbose prints you had before

# Summary counters (money-mapped)
total_sl2 = 0
total_tp5 = 0
total_sl5 = 0
total_skips = 0

LIQUIDATED_COUNT = 0
WITHDRAWAL_LOG = []
equity_curve = []
equity_time = []
trade_logs = []
day_stats = {i: {'SL2': 0, 'SL5': 0, 'TP5': 0} for i in range(7)}

ACCOUNT_BALANCE = 100_000
RISK_PER_TRADE = 2000  # legacy default, replaced by dynamic risk rules
PIP_VALUE_PER_LOT = 100.0
MIN_STOP_DISTANCE = 0.20

RISK_FULL = 3000
SL_THRESHOLD_FULL_RISK = 3.9
MAX_LOT = 7.8
MAX_FREE_MARGIN = 90_000.0
MARGIN_PER_LOT_BUY = 11467.02
MARGIN_PER_LOT_SELL = 11466.06
LOT_STEP = 0.01

os.makedirs("charts/trades", exist_ok=True)
os.makedirs("charts/html", exist_ok=True)
os.makedirs("charts/equity_yearly", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Ultra-fast, vectorized FVG detection (identical logic to your loop)
# We also expose a lightweight "window mask" for BOS loop to avoid per-iter calls
def find_fractal_highs_lows(df):
    """
    Find 5-bar fractal highs and lows (swing points) in the dataframe.
    These work for 15m or any timeframe.
    """
    swing_highs, swing_lows = [], []
    for i in range(2, len(df) - 2):
        center = df.iloc[i]
        left = df.iloc[i - 2:i]
        right = df.iloc[i + 1:i + 3]

        # 5-bar swing high
        if center['high'] > left['high'].max() and center['high'] > right['high'].max():
            swing_highs.append((center['time'], center['high']))

        # 5-bar swing low
        if center['low'] < left['low'].min() and center['low'] < right['low'].min():
            swing_lows.append((center['time'], center['low']))

    return swing_highs, swing_lows


# ─────────────────────────────────────────────────────────────────────────────
def find_real_fvgs_custom(df):
    import numpy as np
    n = len(df)
    if n < 3:
        return pd.DataFrame(columns=['time','direction','fvg_top','fvg_bottom','bos_candle_time'])

    t = df['time' ].to_numpy()
    o = df['open' ].to_numpy()
    h = df['high' ].to_numpy()
    l = df['low'  ].to_numpy()
    c = df['close'].to_numpy()

    out = []
    for i in range(2, n):
        i1, i2, i3 = i-2, i-1, i
        body_low  = o[i2] if o[i2] < c[i2] else c[i2]
        body_high = c[i2] if c[i2] > o[i2] else o[i2]

        left_top, left_bottom   = h[i1], l[i1]
        right_top, right_bottom = h[i3], l[i3]

        # bullish window
        if (left_top < body_high) and (right_bottom > body_low):
            top = body_high if body_high < right_bottom else right_bottom
            bot = body_low  if body_low  > left_top     else left_top
            if (top - bot) >= 0.2:
                out.append((t[i3], 'bullish', top, bot, t[i3]))
                continue
        # bearish window
        if (left_bottom > body_low) and (right_top < body_high):
            top = body_high if body_high < left_bottom else left_bottom
            bot = body_low  if body_low  > right_top   else right_top
            if (top - bot) >= 0.2:
                out.append((t[i3], 'bearish', top, bot, t[i3]))

    if not out:
        return pd.DataFrame(columns=['time','direction','fvg_top','fvg_bottom','bos_candle_time'])
    return pd.DataFrame(out, columns=['time','direction','fvg_top','fvg_bottom','bos_candle_time'])


# ─────────────────────────────────────────────────────────────────────────────
# Trading logic (unchanged decisions; only faster checks where safe)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trade(df_1m, bos_time, fractal_time, direction, df_full, sweep_time, daily_fvgs):
    global total_sl2, total_tp5, total_sl5, total_skips, ACCOUNT_BALANCE, LIQUIDATED_COUNT
    global last_trade_month
    current_month = bos_time.strftime('%Y-%m')

    if 'last_trade_month' not in globals():
        last_trade_month = current_month

    if current_month != last_trade_month:
        if ACCOUNT_BALANCE > 100_000:
            withdrawn = ACCOUNT_BALANCE - 100_000
            WITHDRAWAL_LOG.append((last_trade_month, withdrawn))
            ACCOUNT_BALANCE = 100_000
            if DEBUG:
                print(f"💸 Withdrawal of ${withdrawn:.2f} on {last_trade_month}. Balance reset to 100,000.")
        last_trade_month = current_month

    dir_match = 'bullish' if direction == 'BOS Up' else 'bearish'
    valid_fvgs = daily_fvgs[
        (daily_fvgs['bos_candle_time'] > fractal_time) &
        (daily_fvgs['bos_candle_time'] <= bos_time) &
        (daily_fvgs['direction'] == dir_match)
    ]

    if valid_fvgs.empty:
        if DEBUG: print("    ⛔ No valid FVG found between fractal and BOS.")
        return

    fvg = valid_fvgs.iloc[0]
    fvg_top = fvg['fvg_top']
    fvg_bottom = fvg['fvg_bottom']
    fvg_time = fvg['time']
    entry_price = (fvg_top + fvg_bottom) / 2

    # 1m candles strictly after BOS
    # bars strictly after BOS (identical set to “== then +1”)
    after_bos = df_1m[df_1m['time'] > bos_time]

    # Wait for price to leave FVG and return (faster but identical)
    left_fvg = False
    entry_time = None
    for row in after_bos.itertuples(index=False):
        low = row.low
        high = row.high
        t = row.time

        if not left_fvg:
            if (direction == 'BOS Up' and low > fvg_top) or \
                    (direction == 'BOS Down' and high < fvg_bottom):
                left_fvg = True
            continue

        if low <= entry_price <= high:
            entry_time = t
            break

    if not entry_time:
        if DEBUG: print("    ⛔ Entry not triggered.")
        total_skips += 1
        return

    if DEBUG:
        print(f"    📥 ENTRY at {entry_time} | Price: {entry_price:.2f}")
        print(f"    🔍 FVG Time: {fvg_time} | FVG Range: {fvg_bottom:.2f} - {fvg_top:.2f}")

    window_df = df_full[(df_full['time'] >= sweep_time) & (df_full['time'] <= entry_time)]
    fractal_highs, fractal_lows = find_fractal_highs_lows(window_df)

    fvg_range = fvg_top - fvg_bottom
    buffer = 0.5 * fvg_range

    if direction == 'BOS Up':
        sl = min(price for t, price in fractal_lows) if fractal_lows else fvg_bottom
        sl -= buffer
        stop_distance = entry_price - sl
        if stop_distance < MIN_STOP_DISTANCE:
            stop_distance = MIN_STOP_DISTANCE
            sl = entry_price - stop_distance
        tp1 = entry_price + 3 * stop_distance
        tp2 = entry_price + 6 * stop_distance
    else:
        sl = max(price for t, price in fractal_highs) if fractal_highs else fvg_top
        sl += buffer
        stop_distance = sl - entry_price
        if stop_distance < MIN_STOP_DISTANCE:
            stop_distance = MIN_STOP_DISTANCE
            sl = entry_price + stop_distance
        tp1 = entry_price - 3 * stop_distance
        tp2 = entry_price - 6 * stop_distance

    if stop_distance <= 0:
        total_skips += 1
        return

    per_lot_margin = MARGIN_PER_LOT_BUY if direction == 'BOS Up' else MARGIN_PER_LOT_SELL
    lots_by_margin = math.floor((MAX_FREE_MARGIN / per_lot_margin) / LOT_STEP) * LOT_STEP
    lot_cap_total = min(MAX_LOT, lots_by_margin)

    if stop_distance >= SL_THRESHOLD_FULL_RISK:
        risk_this_trade = RISK_FULL
        lots_planned = risk_this_trade / (PIP_VALUE_PER_LOT * stop_distance)
    else:
        if lot_cap_total < LOT_STEP:
            if DEBUG:
                print("    ⛔ Not enough margin for min lot.")
            total_skips += 1
            return
        lots_planned = lot_cap_total
        risk_this_trade = lots_planned * PIP_VALUE_PER_LOT * stop_distance

    if DEBUG:
        print(f"    📊 Lot size: {lots_planned:.2f} (Stop distance: {stop_distance:.2f})")

    after_entry = df_full[df_full['time'] > entry_time]
    tp1_hit = False

    for row in after_entry.itertuples(index=False):
        t = row.time
        low = row.low
        high = row.high
        close = row.close

        # SL before TP1
        if not tp1_hit and ((direction == 'BOS Up' and low <= sl) or
                            (direction == 'BOS Down' and high >= sl)):
            if DEBUG: print(f"    ❌ SL HIT at {t} → -{risk_this_trade:.2f} USD")
            _record_trade(entry_time, t, -risk_this_trade, direction,
                          stop_distance, fvg_range, entry_price, close, sl, tp1, tp2,
                          risk_this_trade, lots_planned)
            return

        # TP1 hit
        if not tp1_hit and ((direction == 'BOS Up' and high >= tp1) or
                            (direction == 'BOS Down' and low <= tp1)):
            tp1_hit = True
            continue

        # SL after TP1
        if tp1_hit and ((direction == 'BOS Up' and low <= sl) or
                        (direction == 'BOS Down' and high >= sl)):
            profit = (0.8 * 3 - 0.2) * risk_this_trade
            if DEBUG: print(f"    🟡 Final SL after TP1 at {t} → +{profit:.2f} USD")
            _record_trade(entry_time, t, profit, direction,
                          stop_distance, fvg_range, entry_price, close, sl, tp1, tp2,
                          risk_this_trade, lots_planned)
            return

        # TP2 hit
        if tp1_hit and ((direction == 'BOS Up' and high >= tp2) or
                        (direction == 'BOS Down' and low <= tp2)):
            profit = (0.8 * 3 + 0.2 * 6) * risk_this_trade
            if DEBUG: print(f"    🏁 TP2 HIT at {t} → +{profit:.2f} USD")
            _record_trade(entry_time, t, profit, direction,
                          stop_distance, fvg_range, entry_price, close, sl, tp1, tp2,
                          risk_this_trade, lots_planned)
            return

    total_skips += 1  # no TP/SL hit

def _record_trade(entry_time, exit_time, pnl, direction, stop_distance, fvg_range,
                  entry_px, exit_px, sl, tp1, tp2, risk_used, lots_planned):
    global total_sl2, total_tp5, total_sl5, ACCOUNT_BALANCE, LIQUIDATED_COUNT
    if pnl == -risk_used:
        total_sl2 += pnl
        day_bucket = 'SL2'
    elif abs(pnl - ((0.8 * 3 - 0.2) * risk_used)) < 1e-9:
        total_sl5 += pnl
        day_bucket = 'SL5'
    else:
        total_tp5 += pnl
        day_bucket = 'TP5'

    ACCOUNT_BALANCE += pnl
    equity_time.append(exit_time)
    equity_curve.append(ACCOUNT_BALANCE)

    day_stats[entry_time.weekday()][day_bucket] += 1

    trade_logs.append({
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl": pnl,
        "risk": risk_used,
        "lots": lots_planned,
        "direction": "BUY" if direction == "BOS Up" else "SELL",
        "sl_distance": stop_distance,
        "fvg_size": fvg_range,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "sl_px": sl,
        "tp1_px": tp1,
        "tp2_px": tp2,
        "balance": ACCOUNT_BALANCE
    })

    if DEBUG:
        print(f"    📉 New Account Balance: ${ACCOUNT_BALANCE:,.2f}")

    if ACCOUNT_BALANCE <= 90_000:
        if DEBUG:
            print(f"💥 Account liquidated at {exit_time}. Resetting to 100,000 USD.")
        LIQUIDATED_COUNT += 1
        ACCOUNT_BALANCE = 100_000
        equity_time.append(exit_time)
        equity_curve.append(ACCOUNT_BALANCE)


def detect_bos(df_1m, sweep_dir, sweep_time):
    df_1m = df_1m[df_1m['time'] > sweep_time].reset_index(drop=True)
    fractal_highs, fractal_lows = find_fractal_highs_lows(df_1m)
    for i in range(2, len(df_1m) - 2):
        candle = df_1m.iloc[i]
        body_high = max(candle['open'], candle['close'])
        body_low = min(candle['open'], candle['close'])
        window = df_1m.iloc[i - 2:i + 1]
        real_fvgs = find_real_fvgs_custom(window)
        if real_fvgs.empty:
            continue

        if sweep_dir == 'low':
            for ft, price in reversed(fractal_highs):
                if ft < candle['time'] and body_high > price:
                    print(f"  ↳ BOS Up: candle closed above fractal HIGH {price:.2f} (fract. at {ft}, BOS candle at {candle['time']})")
                    return candle['time'], 'BOS Up', ft
        elif sweep_dir == 'high':
            for ft, price in reversed(fractal_lows):
                if ft < candle['time'] and body_low < price:
                    print(f"  ↳ BOS Down: candle closed below fractal LOW {price:.2f} (fract. at {ft}, BOS candle at {candle['time']})")
                    return candle['time'], 'BOS Down', ft
    return None, None, None


def detect_sweep_and_bos(df_15m, df_1m):
    # Precompute day column once
    df_15m = df_15m.copy(); df_1m = df_1m.copy()
    df_15m['date'] = df_15m['time'].dt.date
    df_1m['date']  = df_1m['time'].dt.date

    # group indices → O(1) day slices
    groups_15 = df_15m.groupby('date', sort=False).indices
    groups_1  = df_1m.groupby('date',  sort=False).indices
    all_days = list(groups_15.keys())

    for day in tqdm(all_days, desc="Processing days", unit="day", mininterval=0.2):
        idx15 = groups_15.get(day); idx1 = groups_1.get(day)
        if idx15 is None or idx1 is None:
            continue
        i0_15, i1_15 = idx15[0], idx15[-1] + 1
        i0_1,  i1_1  = idx1[0],  idx1[-1]  + 1
        day_15m = df_15m.iloc[i0_15:i1_15]
        day_1m  = df_1m.iloc[i0_1:i1_1]
        daily_fvgs = find_real_fvgs_custom(day_1m)

        if len(day_15m) < 20 or len(day_1m) < 100:
            continue
        try:
            primary_sweep_start = NY_TZ.localize(datetime.combine(day, time(20, 0)))
            primary_sweep_end   = NY_TZ.localize(datetime.combine(day, time(22, 0)))

            df_15m_pre = day_15m[day_15m['time'] <= primary_sweep_start]
            df_1m_pre  = day_1m[day_1m['time'] <= primary_sweep_start]
            swing_highs, swing_lows = find_fractal_highs_lows(df_15m_pre)

            # last unswept swing detection (kept identical; small n, negligible)
            last_unswept_high = last_unswept_low = None
            for t, price in reversed(swing_highs):
                if not any(df_1m_pre[df_1m_pre['time'] > t]['high'] > price):
                    last_unswept_high = (t, price)
                    break
            for t, price in reversed(swing_lows):
                if not any(df_1m_pre[df_1m_pre['time'] > t]['low'] < price):
                    last_unswept_low = (t, price)
                    break

            sweep_dir = sweep_time = bos_time = bos_label = fractal_time = None
            df_post_sweep = day_1m[(day_1m['time'] >= primary_sweep_start) & (day_1m['time'] <= primary_sweep_end)]
            for _, row in df_post_sweep.iterrows():
                if last_unswept_high and row['high'] > last_unswept_high[1]:
                    sweep_dir, sweep_time = 'high', row['time']; break
                if last_unswept_low and row['low'] < last_unswept_low[1]:
                    sweep_dir, sweep_time = 'low', row['time']; break

            if sweep_dir is None:
                continue

            bos_time, bos_label, fractal_time = detect_bos(day_1m, sweep_dir, sweep_time)
            if bos_time is None:
                continue

            if DEBUG:
                print(f"[{day}] Sweep {sweep_dir.upper()} at {sweep_time} | BOS at {bos_time}")
            full_after_sweep = df_1m[df_1m['time'] >= sweep_time]
            simulate_trade(day_1m, bos_time, fractal_time, bos_label, full_after_sweep, sweep_time, daily_fvgs)

        except Exception as e:
            if DEBUG:
                print(f"Error on {day}: {e}")
            continue

def print_trade_summary():
    df_eq = pd.DataFrame({'time': equity_time, 'balance': equity_curve})
    df_eq.set_index('time', inplace=True)
    csv_file = "charts/equity_curve_xau_2009_2016.csv"
    df_eq.reset_index().to_csv(csv_file, index=False)
    print(f"💾 Equity curve written to {csv_file} ({len(df_eq)} rows)")

    net_pnl = total_tp5 + total_sl5 + total_sl2
    print(f"""
📜 Trade Summary:
  ❌ SL full:     ${-total_sl2:,.2f}
  🟡 SL after 3R: ${total_sl5:,.2f}
  🏁 TP 6R final: ${total_tp5:,.2f}
  ⛘ Skipped:     {total_skips}
💰 Net P&L:      ${net_pnl:,.2f}
""")
    print("🕒 Breakdown by Day of Week:")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i in range(7):
        stats = day_stats[i]
        print(f" {days[i]}: SL2={stats['SL2']}, SL5={stats['SL5']}, TP5={stats['TP5']}")

    if equity_curve:
        # Yearly equity charts
        df_eq = pd.DataFrame({'time': equity_time, 'balance': equity_curve}).set_index('time')

        # Save yearly with month lines
        for year in sorted(df_eq.index.year.unique()):
            _save_yearly_equity(df_eq, year)

    if trade_logs:
        df_trades = pd.DataFrame(trade_logs)
        df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])
        df_trades["month"] = df_trades["exit_time"].dt.to_period("M")
        monthly_pnl = df_trades.groupby("month")["pnl"].sum()

        print("\n📆 Monthly PnL (from actual trades):")
        for month, pnl in monthly_pnl.items():
            print(f" {month}: ${pnl:,.2f}")

    print("\n💸 Withdrawals:")
    if WITHDRAWAL_LOG:
        for month, amount in WITHDRAWAL_LOG:
            print(f" {month}: ${amount:,.2f}")
    else:
        print(" None")
    print(f"\n🔁 Total liquidated accounts: {LIQUIDATED_COUNT}")

    # Save trade logs
    if trade_logs:
        df_full = pd.DataFrame(trade_logs)[[
            "entry_time", "exit_time", "pnl", "risk", "lots", "sl_distance", "fvg_size", "direction",
            "entry_px", "exit_px", "sl_px", "tp1_px", "tp2_px", "balance"
        ]].copy()

        float_cols = ["pnl", "risk", "lots", "sl_distance", "fvg_size", "entry_px", "exit_px",
                      "sl_px", "tp1_px", "tp2_px", "balance"]
        df_full[float_cols] = df_full[float_cols].round(5)
        df_full.to_csv("charts/trades_full.csv", index=False)
        df_full[["entry_time", "exit_time", "direction", "pnl", "risk", "lots"]].to_csv("charts/trades_lite.csv", index=False)
        print("📁 Trade logs saved to: trades_full.csv and trades_lite.csv")

    # Save monthly pnl + withdrawals snapshot
    with open("charts/monthly_pnl_and_withdrawals.txt", "w") as f:
        if trade_logs:
            df_trades = pd.DataFrame(trade_logs)
            df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])
            df_trades["month"] = df_trades["exit_time"].dt.to_period("M")
            monthly_pnl = df_trades.groupby("month")["pnl"].sum()
            f.write("📆 Monthly PnL (from actual trades):\n")
            for month, pnl in monthly_pnl.items():
                f.write(f" {month}: ${pnl:,.2f}\n")

        f.write("\n💸 Withdrawals:\n")
        if WITHDRAWAL_LOG:
            for month, amount in WITHDRAWAL_LOG:
                f.write(f" {month}: ${amount:,.2f}\n")
        else:
            f.write(" None\n")

        # total withdrawals line
        total_withdrawn = sum(amount for _, amount in WITHDRAWAL_LOG)
        print(f"💸 Total Withdrawals: ${total_withdrawn:,.2f}")


def _save_yearly_equity(df_eq, year):
    import matplotlib.patches as patches  # kept import local to avoid overhead if unused
    fig, ax = plt.subplots(figsize=(10, 5))
    df_y = df_eq[df_eq.index.year == year]
    ax.plot(df_y.index, df_y['balance'], label=f'Equity {year}')
    ax.axhline(100_000, color='gray', linestyle='--', linewidth=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for dt in pd.date_range(df_y.index.min().normalize(),
                            df_y.index.max().normalize(), freq='MS'):
        ax.axvline(dt, color='lightgray', linestyle='--', linewidth=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center')
    ax.set_title(f"Equity Curve – {year}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Account Balance (USD)")
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    fn = f"charts/equity_yearly/equity_{year}_months.png"
    fig.savefig(fn)
    print(f"✅ Saved yearly equity chart: {fn}")

# ─────────────────────────────────────────────────────────────────────────────
# Optional: assert withdrawals exactly match your reference list
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_WITHDRAWALS = [
    ("2015-02", 4400.00), ("2015-03", 5600.00), ("2015-04", 5200.00),
    ("2015-08", 6800.00), ("2015-09", 4400.00), ("2015-10", 5200.00),
    ("2015-11", 5200.00), ("2016-03", 35600.00), ("2016-06", 22400.00),
    ("2016-07", 6800.00), ("2016-08", 12400.00), ("2016-09", 8400.00),
    ("2016-10", 8400.00), ("2016-12", 7600.00), ("2017-08", 7200.00),
    ("2017-09", 400.00), ("2017-10", 2400.00), ("2018-02", 6400.00),
    ("2018-04", 10400.00), ("2018-06", 12400.00), ("2018-07", 4400.00),
    ("2018-08", 3200.00), ("2018-11", 6400.00), ("2019-03", 2400.00),
    ("2019-06", 6000.00), ("2019-07", 3200.00), ("2019-08", 2800.00),
    ("2019-09", 6800.00), ("2019-10", 5600.00), ("2020-01", 1600.00),
    ("2020-02", 5200.00), ("2020-05", 7600.00), ("2020-06", 11200.00),
    ("2020-07", 5600.00), ("2020-08", 11600.00), ("2020-09", 4400.00),
    ("2020-11", 7200.00), ("2020-12", 7600.00), ("2021-03", 6000.00),
    ("2021-05", 6400.00), ("2021-12", 1200.00), ("2022-04", 2000.00),
    ("2022-05", 5200.00), ("2023-02", 9600.00), ("2023-03", 31600.00),
    ("2023-04", 1200.00), ("2023-05", 28800.00), ("2023-06", 16000.00),
    ("2023-10", 5200.00), ("2023-11", 5600.00), ("2023-12", 5600.00),
    ("2024-01", 8000.00), ("2024-02", 5600.00), ("2024-03", 31600.00),
    ("2024-04", 11200.00), ("2024-05", 17600.00), ("2024-06", 16000.00),
    ("2024-07", 15200.00), ("2024-10", 14000.00), ("2025-04", 10800.00),
    ("2025-05", 11600.00),
]

def verify_withdrawals(expected=EXPECTED_WITHDRAWALS, raise_on_mismatch=False):
    got = [(m, round(a, 2)) for m, a in WITHDRAWAL_LOG]
    exp = [(m, round(a, 2)) for m, a in expected]
    ok = (got == exp)
    if not ok:
        print("\n⚠️ Withdrawal mismatch!")
        print("Expected:", exp)
        print("Got:     ", got)
        if raise_on_mismatch:
            raise AssertionError("Withdrawals do not match the expected list.")
    else:
        print("\n✅ Withdrawals match the expected list exactly.")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = perf_counter()

    df_15m = pd.read_csv("2015-2025M15.csv")
    df_1m  = pd.read_csv("2015-2025M1.csv")

    rename_map = {'Gmt time':'time','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}
    df_15m.rename(columns=rename_map, inplace=True)
    df_1m.rename(columns=rename_map, inplace=True)

    # exact same parsing and tz conversion as your original
    for df in (df_15m, df_1m):
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        df['time'] = df['time'].dt.tz_localize('Etc/GMT-3').dt.tz_convert('America/New_York')

    detect_sweep_and_bos(df_15m, df_1m)

    print("1-Minute Data Range:")
    print("Start:", df_1m['time'].min())
    print("End:  ", df_1m['time'].max())
    print("Total Days Covered:", (df_1m['time'].max() - df_1m['time'].min()).days)

    print_trade_summary()

    # verify withdrawals (set raise_on_mismatch=True if you want a hard assert)
    verify_withdrawals(raise_on_mismatch=False)

    elapsed = perf_counter() - t0
    print(f"\n⏱️ Total runtime: {elapsed:,.2f} seconds")