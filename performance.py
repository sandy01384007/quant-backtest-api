"""
performance.py — 绩效分析与可视化

计算全套回测指标并生成图表：
  - 年化收益、累计收益、超额收益
  - 夏普比率、索提诺比率、卡玛比率
  - 最大回撤、回撤持续期
  - 胜率、盈亏比、平均持仓天数
  - 月度收益热力图
  - 资金曲线 + 回撤曲线 + 交易明细
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
#  指标计算
# ============================================================
def calc_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame,
                 initial_capital: float, benchmark: pd.Series = None) -> dict:
    """
    计算全套绩效指标

    返回 dict 包含所有核心指标
    """
    equity = equity_curve["equity"]
    returns = equity.pct_change().dropna()

    # ---- 收益类 ----
    total_return = (equity.iloc[-1] / initial_capital - 1)
    trading_days = len(equity)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

    # ---- 风险类 ----
    daily_std = returns.std()
    annual_std = daily_std * np.sqrt(252)
    downside_std = returns[returns < 0].std() * np.sqrt(252)

    # 夏普（无风险利率默认 2%）
    rf = 0.02
    sharpe = (annual_return - rf) / annual_std if annual_std > 0 else 0
    sortino = (annual_return - rf) / downside_std if downside_std > 0 else 0

    # 最大回撤
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = drawdown.min()

    # 卡玛比率
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

    # ---- 交易类 ----
    if not trades.empty:
        # 按卖出计算每笔盈亏
        sells = trades[trades["direction"] == "SELL"].copy()
        buys = trades[trades["direction"] == "BUY"].copy()

        win_trades = 0
        loss_trades = 0
        trade_pnls = []

        # 简化：按顺序配对买卖
        buy_stack = []
        for _, trade in trades.iterrows():
            if trade["direction"] == "BUY":
                buy_stack.append({"price": trade["price"], "shares": trade["shares"], "date": trade.name})
            else:
                remaining = trade["shares"]
                while remaining > 0 and buy_stack:
                    buy = buy_stack[0]
                    matched = min(buy["shares"], remaining)
                    pnl = (trade["price"] - buy["price"]) * matched
                    trade_pnls.append(pnl)
                    if pnl > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    buy["shares"] -= matched
                    remaining -= matched
                    if buy["shares"] <= 0:
                        buy_stack.pop(0)

        total_trades = win_trades + loss_trades
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
        avg_loss = abs(np.mean([p for p in trade_pnls if p <= 0])) if any(p <= 0 for p in trade_pnls) else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")
        expectancy = np.mean(trade_pnls) if trade_pnls else 0
    else:
        total_trades = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        expectancy = 0

    # 基准对比
    benchmark_return = None
    alpha = None
    if benchmark is not None and len(benchmark) > 1:
        benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1)
        benchmark_annual = (1 + benchmark_return) ** (252 / trading_days) - 1
        alpha = annual_return - benchmark_annual

    return {
        "initial_capital": round(initial_capital, 2),
        "final_equity": round(equity.iloc[-1], 2),
        "total_return": round(total_return * 100, 2),
        "annual_return": round(annual_return * 100, 2),
        "annual_std": round(annual_std * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_drawdown * 100, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "trading_days": trading_days,
        "benchmark_return": round(benchmark_return * 100, 2) if benchmark_return is not None else None,
        "alpha": round(alpha * 100, 2) if alpha is not None else None,
    }


# ============================================================
#  可视化
# ============================================================
def plot_results(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict,
    output_path: str,
    strategy_name: str = "Strategy",
    benchmark: pd.Series = None,
):
    """生成回测结果图表（4 合 1）"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    equity = equity_curve["equity"]
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax * 100

    # ---- 1. 资金曲线 ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, color="#185FA5", linewidth=1.5, label="策略净值")
    if benchmark is not None:
        bench_aligned = benchmark.reindex(equity.index).fillna(method="ffill")
        bench_norm = bench_aligned / bench_aligned.iloc[0] * equity.iloc[0]
        ax1.plot(bench_norm.index, bench_norm.values, color="#888780", linewidth=1, label="基准", alpha=0.7)
    ax1.set_title(f"{strategy_name} - 资金曲线", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ---- 2. 回撤曲线 ----
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#E24B4A", alpha=0.4)
    ax2.plot(drawdown.index, drawdown.values, color="#A32D2D", linewidth=0.8)
    ax2.set_title(f"回撤曲线（最大回撤: {metrics['max_drawdown']}%）", fontsize=13, fontweight="bold")
    ax2.set_ylabel("回撤 %")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ---- 3. 月度收益热力图 ----
    ax3 = fig.add_subplot(gs[2, 0])
    monthly = equity.resample("ME").last().pct_change().dropna()
    if len(monthly) > 0:
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values * 100,
        })
        pivot = monthly_df.pivot_table(index="year", columns="month", values="return", aggfunc="first")
        im = ax3.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                        vmin=-10, vmax=10)
        ax3.set_xticks(range(12))
        ax3.set_xticklabels([f"{i+1}月" for i in range(12)], fontsize=9)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(12):
                if j < pivot.shape[1] and not pd.isna(pivot.values[i, j]):
                    ax3.text(j, i, f"{pivot.values[i, j]:.1f}",
                            ha="center", va="center", fontsize=7)
        ax3.set_title("月度收益热力图 (%)", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("月度收益热力图", fontsize=13)

    # ---- 4. 核心指标卡片 ----
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    metric_text = (
        f"  核心绩效指标\n"
        f"{'='*40}\n"
        f"  累计收益:    {metrics['total_return']:>8.2f}%\n"
        f"  年化收益:    {metrics['annual_return']:>8.2f}%\n"
        f"  年化波动:    {metrics['annual_std']:>8.2f}%\n"
        f"  夏普比率:    {metrics['sharpe']:>8.3f}\n"
        f"  索提诺比率:  {metrics['sortino']:>8.3f}\n"
        f"  卡玛比率:    {metrics['calmar']:>8.3f}\n"
        f"  最大回撤:    {metrics['max_drawdown']:>8.2f}%\n"
        f"{'-'*40}\n"
        f"  总交易次数:  {metrics['total_trades']:>8d}\n"
        f"  胜率:        {metrics['win_rate']:>8.1f}%\n"
        f"  盈亏比:      {metrics['profit_factor']:>8.2f}\n"
        f"  单笔期望:    {metrics['expectancy']:>8.2f}\n"
    )
    if metrics.get("benchmark_return") is not None:
        metric_text += f"{'-'*40}\n  基准收益:    {metrics['benchmark_return']:>8.2f}%\n  超额收益:    {metrics['alpha']:>8.2f}%\n"
    ax4.text(0.05, 0.95, metric_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top",
             fontfamily="SimHei",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1EFE8", edgecolor="#D3D1C7"))

    fig.suptitle(f"量化策略回测报告 — {strategy_name}", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[图表] 已保存: {output_path}")


def plot_trade_distribution(trades: pd.DataFrame, output_path: str, strategy_name: str = ""):
    """交易盈亏分布图"""
    if trades.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 配对计算每笔盈亏
    buy_stack = []
    pnls = []
    for _, trade in trades.iterrows():
        if trade["direction"] == "BUY":
            buy_stack.append({"price": trade["price"], "shares": trade["shares"]})
        else:
            remaining = trade["shares"]
            while remaining > 0 and buy_stack:
                buy = buy_stack[0]
                matched = min(buy["shares"], remaining)
                pnls.append((trade["price"] - buy["price"]) / buy["price"] * 100)
                buy["shares"] -= matched
                remaining -= matched
                if buy["shares"] <= 0:
                    buy_stack.pop(0)

    if pnls:
        # 盈亏分布直方图
        colors = ["#E24B4A" if p < 0 else "#1D9E75" for p in pnls]
        axes[0].bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
        axes[0].axhline(y=0, color="#2C2C2A", linewidth=0.8)
        axes[0].set_title("逐笔交易盈亏 (%)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("交易序号")
        axes[0].set_ylabel("收益率 %")
        axes[0].grid(True, alpha=0.3)

        # 累计盈亏曲线
        cum_pnl = np.cumsum(pnls)
        axes[1].plot(range(len(cum_pnl)), cum_pnl, color="#534AB7", linewidth=1.5)
        axes[1].fill_between(range(len(cum_pnl)), cum_pnl, 0,
                             where=cum_pnl >= 0, color="#1D9E75", alpha=0.3)
        axes[1].fill_between(range(len(cum_pnl)), cum_pnl, 0,
                             where=cum_pnl < 0, color="#E24B4A", alpha=0.3)
        axes[1].set_title("累计交易盈亏 (%)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("交易序号")
        axes[1].set_ylabel("累计收益率 %")
        axes[1].grid(True, alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "无已完成交易", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(f"{strategy_name} - 交易分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[图表] 已保存: {output_path}")


def save_summary(metrics: dict, output_path: str):
    """保存指标摘要为 JSON"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[摘要] 已保存: {output_path}")
