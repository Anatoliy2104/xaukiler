import os
from pathlib import Path
from datetime import datetime, time, timedelta

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import plotly.graph_objects as go

from fractal import find_fractal_highs_lows

NY_TZ = pytz.timezone("America/New_York")

DEBUG = False
VERBOSE = False

# Summary counters (money-mapped)
total_sl2 = 0
total_tp5 = 0
total_sl5 = 0
total_skips = 0


LIQUIDATED_COUNT = 0
WITHDRAWAL_LOG = []
equity_curve = []
equity_time = []
all_trades = []    # will hold dicts like {'exit_time': ..., 'pnl': ...}
# ─── TRADE LOGGING ───────────────────────────────────────────────────────────
# will collect one dict per closed trade, then single CSV write at end
trade_logs = []
SKIP_WEEKDAYS = {3}
# Day-of-week stats
day_stats = {i: {'SL2': 0, 'SL5': 0, 'TP5': 0} for i in range(7)}

ACCOUNT_BALANCE = 200_000
RISK_PER_TRADE = 3000
PIP_VALUE_PER_LOT = 1.0

# ─── PROGRESS TRACKING ──────────────────────────────────────────────────────
PROGRESS_PRINT_EVERY_DAYS = 25
PROGRESS_PRINT_EVERY_SECONDS = 20
START_TS = None
ELIGIBLE_DATES = []
ELIGIBLE_TOTAL = 0
eligible_done = 0
trades_closed_done = 0
last_progress_log_ts = None


os.makedirs("charts/trades", exist_ok=True)
os.makedirs("charts/html", exist_ok=True)
os.makedirs("charts/equity_yearly", exist_ok=True)


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_and_cache(csv_path: str, cache_name: str) -> pd.DataFrame:
    """Load CSV data with heavy parsing once, then reuse a cached binary."""
    cache_file = CACHE_DIR / cache_name
    if cache_file.exists():
        df = pd.read_pickle(cache_file)
    else:
        df = pd.read_csv(csv_path)
        rename_map = {
            'Gmt time': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df.rename(columns=rename_map, inplace=True)
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
        df['time'] = df['time'].dt.tz_localize('Etc/GMT-3').dt.tz_convert('America/New_York')
        df['date'] = df['time'].dt.normalize()
        df['weekday'] = df['time'].dt.weekday
        df.to_pickle(cache_file)
    return df


def build_daily_map(df: pd.DataFrame) -> dict:
    """Pre-group DataFrame rows by their date for O(1) day access."""
    return {d: g.reset_index(drop=True) for d, g in df.groupby('date')}


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
'''def plot_trade_chart_interactive(df, entry_time, exit_time, entry_price, sl, tp1, tp2, result_label):
    df_plot = df[(df['time'] >= entry_time - timedelta(minutes=60)) &
                 (df['time'] <= exit_time + timedelta(minutes=30))].copy()

    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['time'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    )])

    y_bottom = min(entry_price, sl, tp2)
    y_top = max(entry_price, sl, tp2)

    fig.add_shape(type="rect",
                  x0=entry_time, x1=exit_time,
                  y0=y_bottom, y1=y_top,
                  line=dict(width=0), fillcolor="lightgray", opacity=0.3)

    for price, name, color in zip([entry_price, sl, tp1, tp2], ['Entry', 'Stop Loss', 'TP1', 'TP2'],
                                  ['blue', 'red', 'orange', 'green']):
        fig.add_shape(type="line",
                      x0=entry_time, x1=exit_time,
                      y0=price, y1=price,
                      line=dict(color=color, dash="dot"))
        fig.add_annotation(x=entry_time, y=price, text=name, showarrow=False,
                           yshift=10 if price < y_top else -10,
                           font=dict(color=color))

    fig.update_layout(title=f"Trade on {entry_time.strftime('%Y-%m-%d %H:%M')} ({result_label})",
                      xaxis_title='Time', yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      template='plotly_white')

    filename = entry_time.strftime("charts/html/%Y-%m-%d_%H-%M.html")
    fig.write_html(filename)'''


def find_real_fvgs_custom(df):
    real_fvgs = []
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c2 = df.iloc[i - 1]
        c3 = df.iloc[i]

        body_low = min(c2['open'], c2['close'])
        body_high = max(c2['open'], c2['close'])

        left_top = c1['high']
        left_bottom = c1['low']
        right_top = c3['high']
        right_bottom = c3['low']

        if left_top < body_high and right_bottom > body_low:
            untouched_top = min(body_high, right_bottom)
            untouched_bottom = max(body_low, left_top)
            if untouched_top - untouched_bottom >= 0.2:
                real_fvgs.append({
                    'time': c3['time'],
                    'direction': 'bullish',
                    'fvg_top': untouched_top,
                    'fvg_bottom': untouched_bottom,
                    'bos_candle_time': c3['time']
                })

        elif left_bottom > body_low and right_top < body_high:
            untouched_top = min(body_high, left_bottom)
            untouched_bottom = max(body_low, right_top)
            if untouched_top - untouched_bottom >= 0.2:
                real_fvgs.append({
                    'time': c3['time'],
                    'direction': 'bearish',
                    'fvg_top': untouched_top,
                    'fvg_bottom': untouched_bottom,
                    'bos_candle_time': c3['time']
                })
    return pd.DataFrame(real_fvgs)


'''def calculate_atr(df, period=10):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr'''


def simulate_trade(df_1m, bos_time, fractal_time, direction, df_full, sweep_time, daily_fvgs):
    global total_sl2, total_tp5, total_sl5, total_skips, ACCOUNT_BALANCE, LIQUIDATED_COUNT, trades_closed_done
    global last_trade_month
    current_month = bos_time.strftime('%Y-%m')

    if 'last_trade_month' not in globals():
        last_trade_month = current_month

    if current_month != last_trade_month:
        if ACCOUNT_BALANCE > 200_000:
            withdrawn = ACCOUNT_BALANCE - 200_000
            WITHDRAWAL_LOG.append((last_trade_month, withdrawn))
            ACCOUNT_BALANCE = 200_000
            if VERBOSE:
                print(f"💸 Withdrawal of ${withdrawn:.2f} on {last_trade_month}. Balance reset to 200,000.")
        last_trade_month = current_month

    dir_match = 'bullish' if direction == 'BOS Up' else 'bearish'
    valid_fvgs = daily_fvgs[
        (daily_fvgs['bos_candle_time'] > fractal_time) &
        (daily_fvgs['bos_candle_time'] <= bos_time) &
        (daily_fvgs['direction'] == dir_match)
        ]

    if valid_fvgs.empty:
        if VERBOSE:
            print("    ⛔ No valid FVG found between fractal and BOS.")
        return

    fvg = valid_fvgs.iloc[0]
    fvg_top = fvg['fvg_top']
    fvg_bottom = fvg['fvg_bottom']
    fvg_time = fvg['time']
    entry_price = (fvg_top + fvg_bottom) / 2

    # Get 1m candles after BOS
    time_to_idx = {t: i for i, t in enumerate(df_1m['time'])}
    bos_idx = time_to_idx.get(bos_time)
    if bos_idx is None:
        return
    after_bos = df_1m.iloc[bos_idx + 1:]

    # Wait for price to leave FVG and return
    left_fvg = False
    entry_time = None
    for row in after_bos.itertuples(index=False):
        if not left_fvg:
            if direction == 'BOS Up' and row.low > fvg_top:
                left_fvg = True
            elif direction == 'BOS Down' and row.high < fvg_bottom:
                left_fvg = True
            continue
        if row.low <= entry_price <= row.high:
            entry_time = row.time
            break

    if not entry_time:
        if VERBOSE:
            print("    ⛔ Entry not triggered.")
        total_skips += 1
        return
    # Get latest ATR before entry
    '''latest_atr_series = df_full[df_full['time'] <= entry_time]['ATR'].dropna()
    if latest_atr_series.empty:
        print(f"    ⛔ No ATR value available for {entry_time}. Skipping trade.")
        total_skips += 1
        return

    latest_atr = latest_atr_series.iloc[-1]

    if latest_atr > 3.5 or latest_atr < 0.15:
        print(f"    ⛔ ATR filter triggered at {entry_time}. ATR={latest_atr:.2f} → Skipping")
        total_skips += 1
        return'''

    if entry_time.hour >= 22:  # or entry_time.weekday() == 3:
        if VERBOSE:
            print(f"    ⛔ Entry at {entry_time} skipped.")
        total_skips += 1
        return

    if VERBOSE:
        print(f"    📥 ENTRY at {entry_time} | Price: {entry_price:.2f}")
        print(f"    🔍 FVG Time: {fvg_time} | FVG Range: {fvg_bottom:.2f} - {fvg_top:.2f}")

    window_df = df_full[(df_full['time'] >= sweep_time) & (df_full['time'] <= entry_time)]
    fractal_highs, fractal_lows = find_fractal_highs_lows(window_df)

    fvg_range = fvg_top - fvg_bottom
    buffer = 0.9 * fvg_range


    if direction == 'BOS Up':
        sl = min(price for t, price in fractal_lows) if fractal_lows else fvg_bottom
        sl -= buffer
        stop_distance = entry_price - sl

        tp1 = entry_price + 3 * stop_distance
        tp2 = entry_price + 6 * stop_distance


    else:
        sl = max(price for t, price in fractal_highs) if fractal_highs else fvg_top
        sl += buffer
        stop_distance = sl - entry_price

        tp1 = entry_price - 3 * stop_distance
        tp2 = entry_price - 6 * stop_distance



    if stop_distance <= 0:
        total_skips += 1
        return

    if stop_distance < 1.67:
        if VERBOSE:
            print(f"    ⛔ SL filter: stop_distance={stop_distance:.2f} < {1.67} → Skipping trade")
        total_skips += 1
        return

    #lot_size = RISK_PER_TRADE / (stop_distance * PIP_VALUE_PER_LOT)
    #print(f"    📊 Lot size: {lot_size:.2f} (Stop distance: {stop_distance:.2f})")

    after_entry = df_full[df_full['time'] > entry_time].reset_index(drop=True)
    tp1_hit = False

    for row in after_entry.itertuples(index=False):
        # 🔻 SL before TP1
        if not tp1_hit and ((direction == 'BOS Up' and row.low <= sl) or
                            (direction == 'BOS Down' and row.high >= sl)):
            if VERBOSE:
                print(f"    ❌ SL HIT at {row.time} → -{RISK_PER_TRADE:.2f} USD")
            total_sl2 += -RISK_PER_TRADE
            ACCOUNT_BALANCE -= RISK_PER_TRADE
            if VERBOSE:
                print(f"    📉 New Account Balance: ${ACCOUNT_BALANCE:,.2f}")
            equity_time.append(row.time)
            equity_curve.append(ACCOUNT_BALANCE)
            day_stats[entry_time.weekday()]['SL2'] += 1
            trade_logs.append({
                "entry_time": entry_time,
                "exit_time": row.time,
                "pnl": -RISK_PER_TRADE,
                "direction": "BUY" if direction == "BOS Up" else "SELL",
                "sl_distance": stop_distance,
                "fvg_size": fvg_range,
                "entry_px": entry_price,
                "exit_px": row.close,
                "sl_px": sl,
                "tp1_px": tp1,
                "tp2_px": tp2,
                "balance": ACCOUNT_BALANCE
            })
            trades_closed_done += 1
            if ACCOUNT_BALANCE <= 180_000 and VERBOSE:
                print(f"💥 Account liquidated at {row.time}. Resetting to 200,000 USD.")
            if ACCOUNT_BALANCE <= 180_000:
                LIQUIDATED_COUNT += 1
                ACCOUNT_BALANCE = 200_000
                equity_time.append(row.time)
                equity_curve.append(ACCOUNT_BALANCE)
            return

        # ✅ TP1 hit
        if not tp1_hit and ((direction == 'BOS Up' and row.high >= tp1) or
                            (direction == 'BOS Down' and row.low <= tp1)):
            tp1_hit = True
            continue

        # 🟡 SL after TP1
        if tp1_hit and ((direction == 'BOS Up' and row.low <= sl) or
                        (direction == 'BOS Down' and row.high >= sl)):
            profit = (0.8 * 3 - 0.2) * RISK_PER_TRADE
            if VERBOSE:
                print(f"    🟡 Final SL after TP1 at {row.time} → +{profit:.2f} USD")
            total_sl5 += profit
            ACCOUNT_BALANCE += profit
            if VERBOSE:
                print(f"    📉 New Account Balance: ${ACCOUNT_BALANCE:,.2f}")

            equity_time.append(row.time)
            equity_curve.append(ACCOUNT_BALANCE)
            day_stats[entry_time.weekday()]['SL5'] += 1
            trade_logs.append({
                "entry_time": entry_time,
                "exit_time": row.time,
                "pnl": profit,
                "direction": "BUY" if direction == "BOS Up" else "SELL",
                "sl_distance": stop_distance,
                "fvg_size": fvg_range,
                "entry_px": entry_price,
                "exit_px": row.close,
                "sl_px": sl,
                "tp1_px": tp1,
                "tp2_px": tp2,
                "balance": ACCOUNT_BALANCE
            })
            trades_closed_done += 1
            if ACCOUNT_BALANCE <= 180_000 and VERBOSE:
                print(f"💥 Account liquidated at {row.time}. Resetting to 200,000 USD.")
            if ACCOUNT_BALANCE <= 180_000:
                LIQUIDATED_COUNT += 1
                ACCOUNT_BALANCE = 200_000
                equity_time.append(row.time)
                equity_curve.append(ACCOUNT_BALANCE)

            return

        # 🏁 TP2 hit
        if tp1_hit and ((direction == 'BOS Up' and row.high >= tp2) or
                        (direction == 'BOS Down' and row.low <= tp2)):

            profit = (0.8 * 3 + 0.2 * 6) * RISK_PER_TRADE
            if VERBOSE:
                print(f"    🏁 TP2 HIT at {row.time} → +{profit:.2f} USD")
            total_tp5 += profit
            ACCOUNT_BALANCE += profit
            if VERBOSE:
                print(f"    📉 New Account Balance: ${ACCOUNT_BALANCE:,.2f}")
            equity_time.append(row.time)
            equity_curve.append(ACCOUNT_BALANCE)
            day_stats[entry_time.weekday()]['TP5'] += 1
            trade_logs.append({
                "entry_time": entry_time,
                "exit_time": row.time,
                "pnl": profit,
                "direction": "BUY" if direction == "BOS Up" else "SELL",
                "sl_distance": stop_distance,
                "fvg_size": fvg_range,
                "entry_px": entry_price,
                "exit_px": row.close,
                "sl_px": sl,
                "tp1_px": tp1,
                "tp2_px": tp2,
                "balance": ACCOUNT_BALANCE
            })
            trades_closed_done += 1
            if ACCOUNT_BALANCE <= 180_000 and VERBOSE:
                print(f"💥 Account liquidated at {row.time}. Resetting to 200,000 USD.")
            if ACCOUNT_BALANCE <= 180_000:
                LIQUIDATED_COUNT += 1
                ACCOUNT_BALANCE = 200_000
                equity_time.append(row.time)
                equity_curve.append(ACCOUNT_BALANCE)

            return

    total_skips += 1  # if no TP/SL hit




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
                    if VERBOSE:
                        print(f"  ↳ BOS Up: candle closed above fractal HIGH {price:.2f} (fract. at {ft}, BOS candle at {candle['time']})")
                    return candle['time'], 'BOS Up', ft
        elif sweep_dir == 'high':
            for ft, price in reversed(fractal_lows):
                if ft < candle['time'] and body_low < price:
                    if VERBOSE:
                        print(f"  ↳ BOS Down: candle closed below fractal LOW {price:.2f} (fract. at {ft}, BOS candle at {candle['time']})")
                    return candle['time'], 'BOS Down', ft
    return None, None, None



def detect_sweep_and_bos(df_15m, df_1m_all, daily_15m, daily_1m):
    global eligible_done, trades_closed_done, last_progress_log_ts, START_TS, ELIGIBLE_TOTAL

    for day, day_15m in daily_15m.items():
        day_1m = daily_1m.get(day)
        if day_1m is None:
            continue

        daily_fvgs = find_real_fvgs_custom(day_1m)

        weekday = day_15m['weekday'].iloc[0]
        if weekday in SKIP_WEEKDAYS:
            continue
        if len(day_15m) < 20 or len(day_1m) < 100:
            continue
        try:
            # Define both sweep windows
            day_date = day.to_pydatetime().date()
            primary_sweep_start = NY_TZ.localize(datetime.combine(day_date, time(20, 00)))
            primary_sweep_end = NY_TZ.localize(datetime.combine(day_date, time(22, 00)))
            #fallback_sweep_start = NY_TZ.localize(datetime.combine(day, time(21, 0)))
            #fallback_sweep_end = NY_TZ.localize(datetime.combine(day, time(23, 0)))

            # ========== PRIMARY WINDOW: 20:00–21:00 ==========

            df_15m_pre = day_15m[day_15m['time'] <= primary_sweep_start]
            df_1m_pre = day_1m[day_1m['time'] <= primary_sweep_start]
            swing_highs, swing_lows = find_fractal_highs_lows(df_15m_pre)

            pre_times = df_1m_pre['time'].to_numpy()
            pre_highs = df_1m_pre['high'].to_numpy()
            pre_lows = df_1m_pre['low'].to_numpy()
            last_unswept_high = last_unswept_low = None
            for t, price in reversed(swing_highs):
                if not (pre_highs[pre_times > t] > price).any():
                    last_unswept_high = (t, price)
                    break
            for t, price in reversed(swing_lows):
                if not (pre_lows[pre_times > t] < price).any():
                    last_unswept_low = (t, price)
                    break

            sweep_dir = sweep_time = bos_time = bos_label = fractal_time = None

            df_post_sweep = day_1m[(day_1m['time'] >= primary_sweep_start) & (day_1m['time'] <= primary_sweep_end)]
            for row in df_post_sweep.itertuples(index=False):
                if last_unswept_high and row.high > last_unswept_high[1]:
                    sweep_dir, sweep_time = 'high', row.time
                    break
                if last_unswept_low and row.low < last_unswept_low[1]:
                    sweep_dir, sweep_time = 'low', row.time
                    break

            # ========== FALLBACK WINDOW: 21:00–23:00 ==========

            '''if sweep_dir is None:
                print(f"⚠️ No sweep from 20:00–21:00 → trying 21:00–23:00")

                df_15m_pre = day_15m[day_15m['time'] <= fallback_sweep_end]
                df_1m_pre = day_1m[day_1m['time'] <= fallback_sweep_end]
                swing_highs, swing_lows = find_fractal_highs_lows(df_15m_pre)

                last_unswept_high = last_unswept_low = None
                for t, price in reversed(swing_highs):
                    if not any(df_1m_pre[df_1m_pre['time'] > t]['high'] > price):
                        last_unswept_high = (t, price)
                        break
                for t, price in reversed(swing_lows):
                    if not any(df_1m_pre[df_1m_pre['time'] > t]['low'] < price):
                        last_unswept_low = (t, price)
                        break

                df_post_sweep = day_1m[
                    (day_1m['time'] >= fallback_sweep_start) & (day_1m['time'] <= fallback_sweep_end)]
                for _, row in df_post_sweep.iterrows():
                    if last_unswept_high and row['high'] > last_unswept_high[1]:
                        sweep_dir, sweep_time = 'high', row['time']
                        break
                    if last_unswept_low and row['low'] < last_unswept_low[1]:
                        sweep_dir, sweep_time = 'low', row['time']
                        break'''

            # ========== BOS detection and trade execution ==========

            if sweep_dir is None:
                continue  # no sweep detected

            bos_time, bos_label, fractal_time = detect_bos(day_1m, sweep_dir, sweep_time)
            if bos_time is None:
                continue  # no BOS detected

            if VERBOSE:
                print(f"[{day_date}] Sweep {sweep_dir.upper()} at {sweep_time} | BOS at {bos_time}")
            full_after_sweep = df_1m_all[df_1m_all['time'] >= sweep_time]
            simulate_trade(day_1m, bos_time, fractal_time, bos_label, full_after_sweep, sweep_time, daily_fvgs)


        except Exception as e:
            if VERBOSE:
                print(f"Error on {day_date}: {e}")
            continue
        finally:
            eligible_done += 1
            now = datetime.now()
            need_print = (eligible_done % PROGRESS_PRINT_EVERY_DAYS == 0) or \
                         ((now - last_progress_log_ts).total_seconds() > PROGRESS_PRINT_EVERY_SECONDS)
            if need_print and ELIGIBLE_TOTAL:
                elapsed = now - START_TS
                elapsed_sec = int(elapsed.total_seconds())
                elapsed_str = str(timedelta(seconds=elapsed_sec))
                remaining_days = ELIGIBLE_TOTAL - eligible_done
                progress_pct = eligible_done / ELIGIBLE_TOTAL * 100
                avg_per_day = elapsed_sec / eligible_done if eligible_done else 0
                eta_seconds = avg_per_day * remaining_days
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                est_trades_total = (trades_closed_done / max(1, eligible_done)) * ELIGIBLE_TOTAL
                est_trades_left = max(0, round(est_trades_total - trades_closed_done))
                print(
                    f"⏱ Progress: {eligible_done:,}/{ELIGIBLE_TOTAL:,} days ({progress_pct:.1f}%) | "
                    f"Trades closed: {trades_closed_done} | Days left: {remaining_days:,} | "
                    f"Elapsed: {elapsed_str} | ETA: {eta_str} | ~Trades left: {est_trades_left} (est)"
                )
                last_progress_log_ts = now
    # Final summary
    if START_TS is not None:
        total_elapsed = datetime.now() - START_TS
        total_elapsed_str = str(timedelta(seconds=int(total_elapsed.total_seconds())))
        print(
            f"✅ Done: {eligible_done:,}/{ELIGIBLE_TOTAL:,} days | Trades closed: {trades_closed_done} | "
            f"Total time: {total_elapsed_str}"
        )


def print_trade_summary():
    df_eq = pd.DataFrame({'time': equity_time, 'balance': equity_curve})
    df_eq.set_index('time', inplace=True)
    # ─── Save raw equity curve for further analysis ──────────────────────────
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

    # ─── Monthly P&L via equity curve ─────────────────────────────────
    if equity_curve:
        # build a DataFrame of balance over time
        df_eq = pd.DataFrame({
            'time': equity_time,
            'balance': equity_curve
        }).set_index('time')

        # get end-of-month balances
        monthly_end = df_eq['balance'].resample('M').last()
        prev_balance = 200_000


        # compute P&L: month’s end minus prior month’s end (first month vs. starting $100 000)
        monthly_pnl = monthly_end.diff().fillna(monthly_end - 200_000)

        #print("\n📊 Monthly P&L:")
        #for dt, pnl in monthly_pnl.items():
        #    print(f" {dt.strftime('%Y-%m')}: ${pnl:,.2f}")

    if equity_curve:
        plt.figure(figsize=(10, 5))
        plt.plot(equity_time, equity_curve, label='Equity Curve')
        plt.axhline(y=200_000, color='gray', linestyle='--', label='Start Balance')
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Account Balance (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()

        # ─── Yearly equity curves with monthly dividers ───────────────────
        # build a DataFrame so we can slice by year
        df_eq = pd.DataFrame({'time': equity_time, 'balance': equity_curve})
        df_eq.set_index('time', inplace=True)

        for year in sorted(df_eq.index.year.unique()):
            df_y = df_eq[df_eq.index.year == year]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_y.index, df_y['balance'], label=f'Equity {year}')
            ax.axhline(200_000, color='gray', linestyle='--', linewidth=1)

            # Major ticks at start of each month, labeled Jan–Dec
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

            # Draw light vertical lines on every month boundary
            for dt in pd.date_range(df_y.index.min().normalize(),
                                    df_y.index.max().normalize(),
                                    freq='MS'):
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
            #plt.show()
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

    if trade_logs:
        df_full = pd.DataFrame(trade_logs)
        df_full["entry_time"] = df_full["entry_time"].dt.strftime("%Y-%m-%d %H:%M%:%S")
        df_full["exit_time"] = df_full["exit_time"].dt.strftime("%Y-%m-%d %H:%M")

        df_full = pd.DataFrame(trade_logs)[[
            "entry_time", "exit_time", "pnl", "sl_distance", "fvg_size", "direction",
            "entry_px", "exit_px", "sl_px", "tp1_px", "tp2_px", "balance"  # 👈 add this
        ]]

        df_lite = df_full[["entry_time", "exit_time", "direction", "pnl"]]
        float_cols = ["pnl", "sl_distance", "fvg_size", "entry_px", "exit_px", "sl_px", "tp1_px", "tp2_px", "balance"]
        df_full[float_cols] = df_full[float_cols].round(5)
        df_full.to_csv("charts/trades_full.csv", index=False)
        df_lite.to_csv("charts/trades_lite.csv", index=False)

        print("📁 Trade logs saved to: trades_full.csv and trades_lite.csv")

    with open("charts/monthly_pnl_and_withdrawals.txt", "w") as f:
        if trade_logs:
            f.write("📆 Monthly PnL (from actual trades):\n")
            for month, pnl in monthly_pnl.items():
                f.write(f" {month}: ${pnl:,.2f}\n")

        f.write("\n💸 Withdrawals:\n")
        if WITHDRAWAL_LOG:
            for month, amount in WITHDRAWAL_LOG:
                f.write(f" {month}: ${amount:,.2f}\n")
        else:
            f.write(" None\n")


if __name__ == '__main__':

    df_15m = load_and_cache("2015-2025M15.csv", "2015-2025M15.pkl")
    df_1m = load_and_cache("2015-2025M1.csv", "2015-2025M1.pkl")

    daily_15m = build_daily_map(df_15m)
    daily_1m = build_daily_map(df_1m)

    START_TS = datetime.now()
    ELIGIBLE_DATES = []
    for day, day_15m in daily_15m.items():
        day_1m = daily_1m.get(day)
        if day_1m is None:
            continue
        weekday = day_15m['weekday'].iloc[0]
        if weekday in SKIP_WEEKDAYS:
            continue
        if len(day_15m) < 20 or len(day_1m) < 100:
            continue
        ELIGIBLE_DATES.append(day)
    ELIGIBLE_TOTAL = len(ELIGIBLE_DATES)
    eligible_done = 0
    trades_closed_done = 0
    last_progress_log_ts = START_TS
    if ELIGIBLE_TOTAL:
        print(
            f"🚀 Backtest started | Eligible days: {ELIGIBLE_TOTAL:,} | From: {min(ELIGIBLE_DATES).date()} to {max(ELIGIBLE_DATES).date()}"
        )
    else:
        print("🚀 Backtest started | Eligible days: 0")

    detect_sweep_and_bos(df_15m, df_1m, daily_15m, daily_1m)
    print("1-Minute Data Range:")
    print("Start:", df_1m['time'].min())
    print("End:  ", df_1m['time'].max())
    print("Total Days Covered:", (df_1m['time'].max() - df_1m['time'].min()).days)

    print_trade_summary()
