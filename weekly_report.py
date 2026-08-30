"""Weekly Telegram performance report for the NIFTY-150 prediction system."""
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from config import PREDICTIONS_CSV
from src.telegram_report import _send

IST = __import__('zoneinfo').ZoneInfo('Asia/Kolkata')
REPORT_PATH = Path('reports/weekly_report.csv')


def _num(x):
    return pd.to_numeric(x, errors='coerce')


def run():
    if not PREDICTIONS_CSV.exists():
        return _send('<b>📊 AI NSE WEEKLY REPORT</b>\nNo prediction ledger available.')

    df = pd.read_csv(PREDICTIONS_CSV)
    if df.empty:
        return _send('<b>📊 AI NSE WEEKLY REPORT</b>\nNo prediction data available.')

    df['target_date'] = pd.to_datetime(df['target_date'], errors='coerce')
    df = df[df['actual_close'].notna()].copy()
    if df.empty:
        return _send('<b>📊 AI NSE WEEKLY REPORT</b>\nNo completed predictions available yet.')

    today = datetime.now(IST).date()
    monday = today - timedelta(days=today.weekday())
    week = df[(df['target_date'].dt.date >= monday) & (df['target_date'].dt.date <= today)].copy()

    # If run on weekend, use the latest completed trading week.
    if week.empty:
        latest = df['target_date'].max().date()
        monday = latest - timedelta(days=latest.weekday())
        week = df[(df['target_date'].dt.date >= monday) & (df['target_date'].dt.date <= latest)].copy()

    cols = ['open','high','low','close']
    for c in cols:
        week[f'{c}_err_pct'] = (week[f'actual_{c}'] - week[f'pred_{c}']).abs() / week[f'actual_{c}'].abs() * 100

    overall_now = week[[f'{c}_err_pct' for c in cols]].mean().mean()
    overall_all = df[[f'{c}_err_pct' for c in cols]].mean().mean() if all(f'{c}_err_pct' in df for c in cols) else None

    # Recompute historical baseline from all completed predictions.
    for c in cols:
        df[f'{c}_err_pct'] = (df[f'actual_{c}'] - df[f'pred_{c}']).abs() / df[f'actual_{c}'].abs() * 100
    overall_all = df[[f'{c}_err_pct' for c in cols]].mean().mean()

    before = None
    if len(df) > len(week):
        before = df[~df.index.isin(week.index)][[f'{c}_err_pct' for c in cols]].mean().mean()

    lines = [
        '<b>📊 AI NSE WEEKLY MODEL REPORT</b>',
        f'Week: <b>{monday.isoformat()} to {today.isoformat()}</b>',
        '',
        '<pre>STOCK          OHLC ERROR %</pre>'
    ]
    stock = week.groupby('symbol')[[f'{c}_err_pct' for c in cols]].mean()
    for symbol, r in stock.sort_values(stock.columns.tolist()).iterrows():
        lines.append(f'<pre>{symbol:<12} {r.mean():>6.2f}%</pre>')

    lines += ['', '<pre>MODEL ACCURACY', '-----------------------------']
    if before is not None:
        lines.append(f'Before current period  {before:.3f}%')
    lines.append(f'Current week           {overall_now:.3f}%')
    lines.append(f'All completed data     {overall_all:.3f}%')
    lines.append(f'Predictions this week  {len(week)}')
    lines.append(f'Completed total        {len(df)}')
    lines.append('</pre>')

    if before is not None:
        change = overall_now - before
        lines.append(f"Accuracy change: <b>{'IMPROVED' if change < 0 else 'WORSENED' if change > 0 else 'UNCHANGED'}</b> ({abs(change):.3f} percentage points)")
    lines.append('<b>Champion replacement: controlled by evening Champion vs Challenger evaluation.</b>')

    REPORT_PATH.parent.mkdir(exist_ok=True)
    week.to_csv(REPORT_PATH, index=False)
    return _send('\n'.join(lines))


if __name__ == '__main__':
    run()
