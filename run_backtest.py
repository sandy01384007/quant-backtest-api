"""
run_backtest.py — 量化回测主运行器

一键运行完整回测流程：
  1. 加载行情数据（真实/合成）
  2. 运行三个实战策略
  3. 输出标准文件：equity.csv / trades.csv / summary.json
  4. 生成可视化图表 PNG
  5. 生成 HTML 综合报告

用法：
    python run_backtest.py                      # 默认：合成数据跑三个策略
    python run_backtest.py --code 600519.SH     # 指定 A 股标的
    python run_backtest.py --code 600519.SH --strategy dual_ma
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest_engine import BacktestEngine
from strategies import (
    DualMA_ATR_Strategy,
    RSI_Reversal_Strategy,
    Donchian_Breakout_Strategy,
)
from data_loader import load_a_share, load_synthetic
from performance import calc_metrics, plot_results, plot_trade_distribution, save_summary


STRATEGY_MAP = {
    "dual_ma": DualMA_ATR_Strategy,
    "rsi": RSI_Reversal_Strategy,
    "donchian": Donchian_Breakout_Strategy,
}


def run_single_strategy(
    df: pd.DataFrame,
    strategy_class,
    symbol: str,
    initial_capital: float = 1_000_000,
    **strategy_kwargs,
) -> dict:
    """运行单个策略回测"""
    strategy_name = strategy_class.__name__
    print(f"\n{'='*60}")
    print(f"  运行策略: {strategy_name}  标的: {symbol}")
    print(f"{'='*60}")

    # 创建引擎并加载数据
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.load_data(symbol, df)

    # 创建并运行策略
    strategy = strategy_class(symbol=symbol, **strategy_kwargs)
    results = engine.run(strategy)

    # 计算绩效指标
    metrics = calc_metrics(
        results["equity_curve"],
        results["trades"],
        initial_capital,
    )

    # 打印核心指标
    print(f"\n  --- 核心指标 ---")
    print(f"  初始资金:    {metrics['initial_capital']:>12,.2f}")
    print(f"  最终权益:    {metrics['final_equity']:>12,.2f}")
    print(f"  累计收益:    {metrics['total_return']:>11.2f}%")
    print(f"  年化收益:    {metrics['annual_return']:>11.2f}%")
    print(f"  年化波动:    {metrics['annual_std']:>11.2f}%")
    print(f"  夏普比率:    {metrics['sharpe']:>12.3f}")
    print(f"  最大回撤:    {metrics['max_drawdown']:>11.2f}%")
    print(f"  总交易次数:  {metrics['total_trades']:>12d}")
    print(f"  胜率:        {metrics['win_rate']:>11.1f}%")
    print(f"  盈亏比:      {metrics['profit_factor']:>12.2f}")

    if not results["trades"].empty:
        print(f"\n  --- 最近 5 笔交易 ---")
        print(results["trades"].tail().to_string())

    return {
        "strategy_name": strategy.name,
        "results": results,
        "metrics": metrics,
    }


def generate_html_report(
    all_results: list[dict],
    output_dir: Path,
    symbol: str,
    data_source: str,
):
    """生成 HTML 综合报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建策略对比卡片
    cards = ""
    for r in all_results:
        m = r["metrics"]
        color = "#1D9E75" if m["total_return"] > 0 else "#E24B4A"
        cards += f"""
        <div class="strategy-card">
            <h3>{r['strategy_name']}</h3>
            <div class="metric-grid">
                <div class="metric"><span class="label">累计收益</span><span class="value" style="color:{color}">{m['total_return']:.2f}%</span></div>
                <div class="metric"><span class="label">年化收益</span><span class="value">{m['annual_return']:.2f}%</span></div>
                <div class="metric"><span class="label">夏普比率</span><span class="value">{m['sharpe']:.3f}</span></div>
                <div class="metric"><span class="label">最大回撤</span><span class="value" style="color:#E24B4A">{m['max_drawdown']:.2f}%</span></div>
                <div class="metric"><span class="label">总交易数</span><span class="value">{m['total_trades']}</span></div>
                <div class="metric"><span class="label">胜率</span><span class="value">{m['win_rate']:.1f}%</span></div>
                <div class="metric"><span class="label">盈亏比</span><span class="value">{m['profit_factor']:.2f}</span></div>
                <div class="metric"><span class="label">卡玛比率</span><span class="value">{m['calmar']:.3f}</span></div>
            </div>
            <img src="{r['strategy_name'].replace('(','').replace(')','').replace('/','_')}_chart.png" alt="chart" onerror="this.style.display='none'">
        </div>
        """

    # 找最佳策略
    best = max(all_results, key=lambda r: r["metrics"]["sharpe"])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化策略回测报告 - {symbol}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; background: #f5f5f5; color: #2c2c2a; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    header {{ background: linear-gradient(135deg, #26215c, #534ab7); color: white; padding: 32px; border-radius: 12px; margin-bottom: 24px; }}
    header h1 {{ font-size: 24px; font-weight: 500; margin-bottom: 8px; }}
    header .meta {{ font-size: 13px; opacity: 0.85; }}
    .summary-bar {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
    .summary-item {{ background: white; padding: 20px; border-radius: 10px; flex: 1; min-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .summary-item .label {{ font-size: 12px; color: #888780; display: block; margin-bottom: 4px; }}
    .summary-item .value {{ font-size: 22px; font-weight: 500; color: #26215c; }}
    .strategy-card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .strategy-card h3 {{ font-size: 16px; color: #26215c; margin-bottom: 16px; border-bottom: 2px solid #eeedfe; padding-bottom: 8px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
    .metric {{ background: #f1efe8; padding: 12px; border-radius: 8px; text-align: center; }}
    .metric .label {{ font-size: 11px; color: #888780; display: block; }}
    .metric .value {{ font-size: 18px; font-weight: 500; color: #2c2c2a; }}
    .strategy-card img {{ width: 100%; border-radius: 8px; margin-top: 12px; }}
    .best-badge {{ display: inline-block; background: #1d9e75; color: white; padding: 2px 10px; border-radius: 12px; font-size: 11px; margin-left: 8px; vertical-align: middle; }}
    footer {{ text-align: center; padding: 24px; color: #888780; font-size: 12px; }}
    .disclaimer {{ background: #faecde; border-left: 4px solid #ba7517; padding: 16px; border-radius: 8px; margin: 20px 0; font-size: 13px; color: #633806; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>量化策略回测报告</h1>
        <div class="meta">标的: {symbol} | 数据源: {data_source} | 生成时间: {timestamp}</div>
    </header>

    <div class="summary-bar">
        <div class="summary-item">
            <span class="label">最佳策略（夏普最高）</span>
            <span class="value">{best['strategy_name']}</span>
        </div>
        <div class="summary-item">
            <span class="label">最佳夏普比率</span>
            <span class="value">{best['metrics']['sharpe']:.3f}</span>
        </div>
        <div class="summary-item">
            <span class="label">最佳年化收益</span>
            <span class="value">{best['metrics']['annual_return']:.2f}%</span>
        </div>
        <div class="summary-item">
            <span class="label">运行策略数</span>
            <span class="value">{len(all_results)}</span>
        </div>
    </div>

    {cards}

    <div class="disclaimer">
        <strong>风险提示：</strong>以上回测结果基于历史数据，不代表未来收益。回测存在过拟合风险，
        且未完全考虑滑点、流动性等真实交易摩擦。本报告仅供学习研究，不构成任何投资建议。
    </div>

    <footer>
        <p>⚠️ 以上内容由 AI 基于公开信息整理生成，仅供参考，不构成任何投资建议或个股推荐。投资有风险，决策需谨慎。</p>
        <p>Generated by quant_backtest framework | {timestamp}</p>
    </footer>
</div>
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n[报告] HTML 报告已生成: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="量化策略回测框架")
    parser.add_argument("--code", default="DEMO", help="标的代码（如 600519.SH），默认合成数据")
    parser.add_argument("--days", type=int, default=500, help="回测数据天数")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--strategy", default="all", choices=["all", "dual_ma", "rsi", "donchian"],
                        help="运行哪个策略")
    parser.add_argument("--output", default="output", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载数据 ----
    print(f"\n{'#'*60}")
    print(f"  量化策略回测框架 v1.0")
    print(f"  标的: {args.code} | 数据量: {args.days} 天 | 初始资金: {args.capital:,.0f}")
    print(f"{'#'*60}")

    if args.code == "DEMO":
        df = load_synthetic("DEMO", args.days, seed=42)
        data_source = "合成数据(GBM)"
        symbol = "DEMO"
    else:
        df = load_a_share(args.code, args.days)
        data_source = "westock-data/yfinance"
        symbol = args.code

    print(f"[数据] 加载完成: {len(df)} 条 | 时间范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    # ---- 选择策略 ----
    if args.strategy == "all":
        strategies_to_run = [
            (DualMA_ATR_Strategy, {}),
            (RSI_Reversal_Strategy, {}),
            (Donchian_Breakout_Strategy, {}),
        ]
    else:
        strategies_to_run = [(STRATEGY_MAP[args.strategy], {})]

    # ---- 运行回测 ----
    all_results = []
    for strat_class, kwargs in strategies_to_run:
        result = run_single_strategy(df, strat_class, symbol, args.capital, **kwargs)
        all_results.append(result)

        # 保存标准文件
        prefix = strat_class.__name__
        result["results"]["equity_curve"].to_csv(output_dir / f"{prefix}_equity.csv")
        result["results"]["trades"].to_csv(output_dir / f"{prefix}_trades.csv")
        save_summary(result["metrics"], output_dir / f"{prefix}_summary.json")

        # 生成图表
        chart_name = prefix.replace("(", "").replace(")", "").replace("/", "_")
        plot_results(
            result["results"]["equity_curve"],
            result["results"]["trades"],
            result["metrics"],
            str(output_dir / f"{chart_name}_chart.png"),
            strategy_name=result["strategy_name"],
        )
        plot_trade_distribution(
            result["results"]["trades"],
            str(output_dir / f"{chart_name}_trades.png"),
            result["strategy_name"],
        )

    # ---- 生成 HTML 报告 ----
    report_path = generate_html_report(all_results, output_dir, symbol, data_source)

    # ---- 总结 ----
    print(f"\n{'='*60}")
    print(f"  回测完成！输出目录: {output_dir.resolve()}")
    print(f"{'='*60}")
    print(f"\n  生成的文件:")
    for f in sorted(output_dir.iterdir()):
        print(f"    - {f.name}")

    print(f"\n  查看报告: {report_path.resolve()}")
    return report_path


if __name__ == "__main__":
    main()
