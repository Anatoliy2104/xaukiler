# Backtest Speed-Up — **Codex Implementation README**

**Goal:** Reduce backtest time from ~1.5h → **≤ 30 min** (CPU) **without changing any strategy logic** or trade outcomes.

This README tells Codex (and any developer) exactly what to change, where, and how to verify that results remain identical.

---

## Repository Context

- **Main script:** `slow.py`
- **Data:** `2015-2025M1.csv`, `2015-2025M15.csv`
- **Outputs:** `charts/` (trade logs, equity, images)
- **Timezone rule:** Load CSV in `Etc/GMT-3`, convert to `America/New_York` (keep identical).

> **Hard rule:** *Do not change trading logic or thresholds.* All modifications are performance-only.

---

## Performance Strategy (No-Logic-Change)

1. **Cache parsed data (first run only; reuse next runs).**
   - Read CSV → rename columns → parse `time` → localize to `Etc/GMT-3` → convert to `America/New_York`.
   - Add `date = time.normalize()` (tz-aware midnight).
   - Save **binary cache** per file (e.g., `2015-2025M1.pkl`, `2015-2025M15.pkl`).  
   - On subsequent runs: **load the pickle** instead of CSV.

2. **Pre-group once for O(1) daily access.**
   - Build maps once: `daily_15m = {date: day_df}`, `daily_1m = {date: day_df}` via `groupby('date')`.
   - In the daily loop use dictionary lookups, not boolean masks like `df['time'].dt.date == day`.

3. **Preserve cross-midnight exits.**
   - Use the **daily** slices only for sweep/BOS detection.
   - For trade simulation after entry, keep a **global `after_sweep` M1 window** (`time >= sweep_time`) so exits can occur after midnight exactly as in `slow.py`.

4. **Use faster row access (same values).**
   - Replace slow row iteration with **tuple-style access** (no logic edits) in:
     - Post-sweep scan (find sweep candle).
     - Post-entry scan (SL/TP1/TP2 event ordering).

5. **Find BOS index within the day slice.**
   - Locate `bos_time` inside the day’s 1m DataFrame (small search).
   - Still use the global `after_sweep` for exits (see #3).

6. **Remove hot-loop I/O.**
   - Silence frequent `print()` in tight loops behind a `VERBOSE` flag (default OFF).
   - Defer all file writes and chart rendering until **after** the simulation finishes.

7. **Micro-optimizations (safe).**
   - Reuse precomputed `date` and `weekday` (first row of the day slice) for skip logic.
   - Avoid creating many temporary DataFrames inside loops when arrays from the slice suffice.

8. **Optional accelerators (only if needed).**
   - Numba-JIT the post-entry event scanner (pure numpy arrays; same thresholds/order).
   - Parallelize **indicator precomputation** per day (not the simulation state machine).

---

## Concrete Tasks for Codex

**T1 — Add a cached loader**  
Create a helper that:
- Accepts (`csv_path`, `pkl_path`, `rename_map`, `src_tz='Etc/GMT-3'`, `dst_tz='America/New_York'`).
- On first run: read CSV, rename columns, parse & tz-convert `time`, set `date = time.normalize()`, then `to_pickle(pkl_path)`.
- On later runs: `read_pickle(pkl_path)`.
- Guarantee `'date'` exists (for old caches).

**T2 — Build daily maps once**  
Create `build_daily_maps(df_15m, df_1m)` to return `{date: df}` dicts using `groupby('date', sort=False)`.

**T3 — Fast sweep/BOS loop**  
Add a function like `detect_sweep_and_bos_fast(daily_15m, daily_1m)` that mirrors `detect_sweep_and_bos` logic but:
- Iterates `for date, day_15m in daily_15m.items()`.
- Fetches `day_1m = daily_1m.get(date)`.
- Uses tuple-style iteration to find the first sweep in the primary window.
- Calls the existing `detect_bos(...)` unchanged.
- Creates `full_after_sweep` from the **global M1 DataFrame** (`time >= sweep_time`).

**T4 — Fast simulate (same logic)**  
Add a `simulate_trade_fast(...)` that is **logic-identical** to `simulate_trade(...)` but:
- Works with day slices for BOS index lookup.
- Uses tuple-style iteration for post-BOS and post-entry scans.
- Keeps the same SL/TP order and account updates, withdrawals, liquidation rules.

**T5 — Main switches to cache + fast path**  
In `__main__`:
- Replace CSV direct loads with **T1 cached loader** for both files.
- Build **T2** maps.
- Call **T3** fast detector instead of the original function.
- Keep the equity/summary/reporting code as is.

**T6 — Logging control**  
Introduce `VERBOSE = False` and route non-critical prints through a `log(...)` helper.  
Keep critical events (`ENTRY`, `SL`, `TP1`, `TP2`, withdrawals, liquidations) printed as before.

**T7 — Avoid equality lookups across 10y data**  
When locating `bos_time`, find it **within the day slice**, not via a global search.

**T8 — Keep cross-midnight behavior**  
Ensure `after_entry` scanning uses the **global** `df_full = df_1m[df_1m['time'] > entry_time]` derived from `time >= sweep_time` (as in `slow.py`).

**T9 — I/O discipline**  
Do not write files or render charts inside the hot loops. Aggregate in-memory and write once at the end (already mostly true).

**T10 — (Optional) Numba on post-entry scanner**  
If needed for further speed, move the event scan (after entry) into a Numba-compiled function that receives arrays of highs/lows (and thresholds) and returns which event hit first and when. Do not change event precedence.

---

## Non‑Negotiables (Do Not Change)

- All thresholds, comparisons, and order of checks (SL before TP1; TP1 gating for TP2/SL-after-TP1) must remain **exactly** as in `slow.py`.
- Monthly withdrawal/reset rules and liquidation threshold must be **identical**.
- Timezone pipeline (CSV tz localize → convert) must be **identical**.
- FVG/fractal detection logic and inputs must be **identical**.
- Exits can happen after midnight (do not cap to a day).

---

## Acceptance Criteria (Regression Gate)

1. **Trade equality**  
   - `trades_full.csv` after the run matches baseline 1:1 on:  
     `entry_time, exit_time, pnl, sl_distance, fvg_size, direction, entry_px, exit_px, sl_px, tp1_px, tp2_px, balance`.

2. **Counts match**  
   - Same number of trades, skips, withdrawals (months and amounts), liquidations (count and timestamps).

3. **Equity curve equality**  
   - `equity_time` and `equity_curve` sequences identical (timestamp-by-timestamp).

4. **Spot checks**  
   - Manually inspect a few days with cross-midnight exits—identical events and ordering.

5. **Performance**  
   - Full 2015–2025 run completes in **≤ 30 minutes** on CPU.

> If any check fails, revert only the last change and re‑test. Do not “fix” by adjusting thresholds or logic.

---

## Runbook

1. **First run (build caches):** run the script once to generate `.pkl` caches from CSVs.  
2. **Subsequent runs:** script should auto‑load `.pkl` and skip CSV parsing/tz conversion.  
3. **Logs & charts:** produced at end as usual under `charts/`.
4. **Toggle verbosity:** set `VERBOSE = True` only when debugging—expect a slowdown.

---

## Notes for Future Work (still logic‑safe)
- If more speed is required, prefer **Numba** on the post-entry scanner before attempting full simulation parallelism (stateful month boundaries make parallelization tricky).  
- If you parallelize anything now, limit it to **indicator precomputation per day** and merge results before simulation.

---

## File Checklist (post‑implementation)

- `slow.py` (updated with cached loader, daily maps, fast detectors)
- `2015-2025M1.pkl`, `2015-2025M15.pkl` (auto-generated on first run)
- `charts/trades_full.csv`, `charts/trades_lite.csv` (unchanged schema)
- `charts/equity_yearly/*.png` (optional during dev; generate on demand)

---

**Owner:** Anatoliy  
**SLO:** ≤ 30 min full backtest, identical outputs to pre‑optimization baseline.
