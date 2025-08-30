# XAUUSD Sweep→BOS→FVG Backtester (FTMO‑aware)

This repository contains a single script, `fast.py`, that backtests a **sweep → BOS (break of structure) → FVG** entry model on **XAU/USD** using 1‑minute and 15‑minute CSV data. It includes **FTMO‑style risk/margin constraints**, monthly withdrawals back to a base balance, and liquidation resets. It saves equity charts, trade logs, and a monthly PnL + withdrawals snapshot for review.

---

## TL;DR

- Put two CSVs in the working folder: `2015-2025M15.csv` and `2015-2025M1.csv` (schema below).
- Run:  
  ```bash
  python fast.py
  ```
- Outputs land in `charts/…`:
  - `charts/trades_full.csv`, `charts/trades_lite.csv`
  - `charts/equity_yearly/equity_<YEAR>_months.png`
  - `charts/equity_curve_xau_2009_2016.csv` (equity timeseries)
  - `charts/monthly_pnl_and_withdrawals.txt`
- Optional: set `DEBUG = True` at the top for verbose trace.

---

## Requirements

- **Python 3.9+**
- Libraries: `pandas`, `pytz`, `matplotlib`, `tqdm` (optional; the script gracefully falls back if missing)

Install:
```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install pandas pytz matplotlib tqdm
# macOS/Linux
source .venv/bin/activate && pip install pandas pytz matplotlib tqdm
```

---

## Data Inputs

The script expects two CSV files in the current directory:

- `2015-2025M15.csv` — 15‑minute bars
- `2015-2025M1.csv` — 1‑minute bars

**Required columns** (exact names):  
`Gmt time, Open, High, Low, Close, Volume`

**Datetime format:** `YYYY-MM-DD HH:MM:SS` (no timezone in the string).  
The script converts times from `Etc/GMT-3` to `America/New_York` internally.

---

## What the Strategy Does (high level)

1. **Day slicing:** For each trading day it prepares 15m and 1m slices.
2. **Find “last unswept” 15m swing** before 20:00 New York.
3. **Primary sweep window:** Detects if that swing is swept **between 20:00 and 22:00 New York** on 1m data.
4. **BOS detection:** After the sweep, finds a 1m candle that closes beyond the *opposite* fractal (with a small look‑back), **and confirms a valid “real FVG.”**
5. **Entry:** Chooses the matching FVG (same direction as BOS). Entry = **midpoint of the FVG** after price first leaves the FVG and then returns. Entries at/after **22:00** are skipped.
6. **Stops & Targets:**
   - **SL** at the nearest opposing fractal ± a buffer (buffer = 90% of the FVG range).
   - **TP1 = 3R**, **TP2 = 6R** (R = risk used on the trade).
   - After TP1, 80% is treated as banked; if SL later hits, net ≈ **+2.2R**; if TP2 hits, net ≈ **+4.2R**.
7. **Sizing & FTMO constraints:**
   - If stop distance ≥ threshold, risk the **full amount** (configurable).
   - Otherwise, cap by **max lot**, **lot step**, and **per‑lot margin**; also respect a **free‑margin cap**.
8. **Accounting model:**
   - **Monthly withdrawal**: on month change, withdraw any balance above **$100,000** and reset to $100k.
   - **Liquidation**: if balance ≤ **$90,000** after a trade, count a liquidation and reset to $100k.

---

## Configuration (edit at the top of `fast.py`)

| Name | Purpose | Default |
|---|---|---|
| `ACCOUNT_BALANCE` | Starting balance | `100_000` |
| `PIP_VALUE_PER_LOT` | Pip value per lot (XAUUSD) | `100.0` |
| `RISK_FULL` | Full risk per trade (used when SL ≥ threshold) | `3000` |
| `SL_THRESHOLD_FULL_RISK` | SL distance threshold to allow full risk | `3.9` |
| `MAX_LOT` | Absolute lot cap | `7.8` |
| `MAX_FREE_MARGIN` | Upper bound used for margin sizing | `90_000.0` |
| `MARGIN_PER_LOT_BUY` | Per‑lot margin (buy) | `11467.02` |
| `MARGIN_PER_LOT_SELL` | Per‑lot margin (sell) | `11466.06` |
| `LOT_STEP` | Lot granularity | `0.01` |
| `DEBUG` | Verbose prints | `False` |

**Time windows & timezone**
- All internal logic is in **America/New_York** time.
- Primary sweep window: **20:00–22:00 NY**.
- No entries after **22:00 NY**.

---

## File Outputs

- **Equity curve CSV**: `charts/equity_curve_xau_2009_2016.csv`
- **Yearly equity plots**: `charts/equity_yearly/equity_<YEAR>_months.png` (month gridlines included)
- **Trades CSVs**:  
  - `charts/trades_full.csv` (all trade fields, including prices, distances, running balance)  
  - `charts/trades_lite.csv` (compact: entry, exit, direction, pnl, risk, lots)
- **Monthly PnL + Withdrawals**: `charts/monthly_pnl_and_withdrawals.txt`
- **Console summary**: SL/TP buckets, skipped, per‑weekday counts, monthly PnL printout, withdrawals list and total, 1m data coverage and runtime.

> Tip: the script also keeps an in‑memory `WITHDRAWAL_LOG` and prints a **Total Withdrawals** figure at the end.

---

## CSV Schema Notes

The script renames columns internally:
```
'Gmt time' -> 'time'
'Open'     -> 'open'
'High'     -> 'high'
'Low'      -> 'low'
'Close'    -> 'close'
'Volume'   -> 'volume'
```
It then parses `'time'` with `format='%Y-%m-%d %H:%M:%S'`, localizes from `Etc/GMT-3`, and converts to `America/New_York`.

---

## Internals (for reviewers)

- **Fractals:** 5‑bar swing highs/lows on the active dataframe (works for 15m or 1m slices).
- **FVG filter:** Custom “real FVG” check over 3 consecutive candles with a **minimum gap size** (default `0.2`). The BOS candle’s mini‑window must include such an FVG; the final trade uses an FVG between the fractal and BOS that matches the BOS direction.
- **Entry rule:** Wait for price to **leave** the FVG, then **return** to the FVG midpoint to fill.
- **Bucketed outcomes:**  
  - SL before TP1 → **SL2** (−1R)  
  - TP1 then SL → **SL5** (+2.2R)  
  - TP1 then TP2 → **TP5** (+4.2R)
- **Accounting:**  
  - **Monthly withdrawals**: on month change, if balance > 100k, withdraw the difference (log month + amount) and set balance back to 100k.  
  - **Liquidations**: if balance ≤ 90k at exit, increment a counter and immediately reset to 100k (equity curve continues).

---

## How to Run

1. Place the two CSVs in the working directory.
2. (Optional) Tweak configuration at the top of `fast.py`.
3. Run:
   ```bash
   python fast.py
   ```
4. Inspect outputs under `charts/` and check the console summary.

---

## Verifying Against a Reference Withdrawals List

At the bottom of the script there’s `EXPECTED_WITHDRAWALS` and a helper `verify_withdrawals(...)`. By default it prints whether the run matches the list. You can set `raise_on_mismatch=True` to hard‑fail the run if they differ.

Example:
```python
verify_withdrawals(raise_on_mismatch=True)
```

If your data source or settings change, update `EXPECTED_WITHDRAWALS` accordingly.

---

## Customization Ideas

- Change the **sweep window** or **entry cutoff** hour.
- Adjust **FVG minimum size**, **buffer %**, or **risk/threshold** parameters.
- Swap per‑lot margin numbers or **max lot** to reflect a different broker/product.
- Log additional diagnostics to `charts/html` or add per‑trade charts.

---

## Troubleshooting

- **“No valid FVG found between fractal and BOS.”**  
  The BOS mini‑window did not meet the FVG criteria; try reducing the FVG size threshold or review the day’s structure.

- **“Entry not triggered.”**  
  Price did not return to the FVG midpoint after leaving it, or it happened after 22:00.

- **“Not enough margin for min lot.”**  
  With the current per‑lot margin and caps, lot size could not reach the minimum `LOT_STEP`. Increase `MAX_FREE_MARGIN`/`MAX_LOT` or use full‑risk mode by increasing the SL threshold and/or typical SL distances.

- **Equity plots look empty/weird.**  
  Check timezone conversions and make sure the CSVs actually cover the 20:00–22:00 New York window.

- **Withdrawal mismatch.**  
  Either update `EXPECTED_WITHDRAWALS` or ensure you’re using the same dataset and parameters as the reference.

---

## License

Proprietary/backtest‑lab use. Adapt as needed within your project.
