from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from trade_lab.backtesting.engine import BacktestResult


def generate_report(
    result: BacktestResult,
    output_path: str = 'backtest_report.html',
) -> str:
    """Generate an HTML backtest report with charts and metrics.

    Parameters
    ----------
    result : BacktestResult
        Output of ``BacktestEngine.run()``.
    output_path : str
        File path for the HTML report.

    Returns
    -------
    str
        The output file path.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=['Equity Curve', 'Drawdown', 'Price & Trades'],
        vertical_spacing=0.06,
        row_heights=[0.4, 0.2, 0.4],
    )

    # ---- Equity curve ----
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index.to_numpy(),
            y=result.equity_curve.to_numpy(),
            mode='lines',
            name='Equity',
            line=dict(color='#2962FF'),
        ),
        row=1, col=1,
    )

    # ---- Drawdown ----
    peak = result.equity_curve.cummax()
    dd = (result.equity_curve - peak) / peak * 100
    fig.add_trace(
        go.Scatter(
            x=dd.index.to_numpy(),
            y=dd.to_numpy(),
            mode='lines',
            fill='tozeroy',
            name='Drawdown %',
            line=dict(color='#FF6D00'),
        ),
        row=2, col=1,
    )

    # ---- Price + trade markers ----
    fig.add_trace(
        go.Scatter(
            x=result.df.index.to_numpy(),
            y=result.df['Close'].to_numpy(),
            mode='lines',
            name='Close',
            line=dict(color='#666'),
        ),
        row=3, col=1,
    )

    if len(result.trade_log) > 0:
        tl = result.trade_log

        longs = tl[tl['direction'] == 'long']
        shorts = tl[tl['direction'] == 'short']

        if len(longs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=longs['entry_date'], y=longs['entry_price'],
                    mode='markers', name='Long Entry',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                ),
                row=3, col=1,
            )
        if len(shorts) > 0:
            fig.add_trace(
                go.Scatter(
                    x=shorts['entry_date'], y=shorts['entry_price'],
                    mode='markers', name='Short Entry',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                ),
                row=3, col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=tl['exit_date'], y=tl['exit_price'],
                mode='markers', name='Exit',
                marker=dict(symbol='x', size=8, color='gray'),
            ),
            row=3, col=1,
        )

    fig.update_layout(height=900, title_text='Backtest Report', showlegend=True)

    # ---- Assemble HTML ----
    metrics_table = _format_metrics_tables(result.metrics)
    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    html = (
        '<!DOCTYPE html>\n<html>\n<head>\n'
        '<meta charset="utf-8">\n'
        '<title>Backtest Report</title>\n'
        '<style>\n'
        'body { font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }\n'
        'h1 { color: #333; }\n'
        '.metrics-grid { '
        'display: grid; '
        'grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); '
        'gap: 20px; margin: 20px 0; '
        '}\n'
        '.metrics-card h2 { margin: 0 0 8px 0; color: #333; font-size: 1.05rem; }\n'
        'table { border-collapse: collapse; width: 100%; background: white; }\n'
        'th, td { border: 1px solid #ddd; padding: 8px 16px; text-align: right; }\n'
        'th { background: #f5f5f5; text-align: left; }\n'
        '.pos { color: #2e7d32; }\n'
        '.neg { color: #c62828; }\n'
        '</style>\n</head>\n<body>\n'
        '<h1>Backtest Report</h1>\n'
        f'{metrics_table}\n'
        f'{chart_html}\n'
        '</body>\n</html>'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_METRIC_FORMAT_MAIN = [
    ('total_return',      'Total Return',       '{:.2%}',      True),
    ('annualized_return', 'Annualized Return',  '{:.2%}',      True),
    ('sharpe_ratio',      'Sharpe Ratio',       '{:.2f}',      True),
    ('sortino_ratio',     'Sortino Ratio',      '{:.2f}',      True),
    ('max_drawdown',      'Max Drawdown',       '{:.2%}',      True),
    ('annual_volatility', 'Annual Volatility',  '{:.2%}',      False),
    ('total_trades',      'Total Trades',       '{:.0f}',      False),
    ('win_rate',          'Win Rate',           '{:.1%}',      False),
    ('profit_factor',     'Profit Factor',      '{:.2f}',      False),
    ('avg_win',           'Avg Win',            '${:,.2f}',    False),
    ('avg_loss',          'Avg Loss',           '${:,.2f}',    False),
    ('avg_trade_bars',    'Avg Trade Duration', '{:.1f} bars', False),
    ('total_commission',  'Total Commission',   '${:,.2f}',    False),
]

_METRIC_FORMAT_DIRECTIONAL = [
    ('long_win_rate',   'Long Win Rate',   '{:.1%}',   False),
    ('long_avg_win',    'Long Avg Profit', '${:,.2f}', True),
    ('long_avg_loss',   'Long Avg Loss',   '${:,.2f}', True),
    ('short_win_rate',  'Short Win Rate',  '{:.1%}',   False),
    ('short_avg_win',   'Short Avg Profit','${:,.2f}', True),
    ('short_avg_loss',  'Short Avg Loss',  '${:,.2f}', True),
]


def _format_metrics_table(metrics: dict, metric_format: list[tuple]) -> str:
    rows: list[str] = []
    for key, label, fmt, colorise in metric_format:
        val = metrics.get(key, 0)
        formatted = fmt.format(val)
        css = ''
        if colorise:
            css = ' class="pos"' if val > 0 else ' class="neg"' if val < 0 else ''
        rows.append(f'<tr><th>{label}</th><td{css}>{formatted}</td></tr>')
    return f'<table>{"".join(rows)}</table>'


def _format_metrics_tables(metrics: dict) -> str:
    main_table = _format_metrics_table(metrics, _METRIC_FORMAT_MAIN)
    directional_table = _format_metrics_table(metrics, _METRIC_FORMAT_DIRECTIONAL)
    return (
        '<div class="metrics-grid">'
        '<section class="metrics-card">'
        '<h2>Overall Metrics</h2>'
        f'{main_table}'
        '</section>'
        '<section class="metrics-card">'
        '<h2>Long / Short Breakdown</h2>'
        f'{directional_table}'
        '</section>'
        '</div>'
    )
