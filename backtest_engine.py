"""
backtest_engine.py — 实战级量化回测引擎

设计要点：
  - 逐 bar 驱动，策略在每个 bar 收盘后生成信号，下一 bar 开盘执行（杜绝前视偏差）
  - 内置 A 股真实交易规则：T+1、100 股最小手数、佣金、印花税、过户费
  - 支持多标的、仓位管理、止损止盈
  - 资金曲线 + 交易明细完整记录

使用方式见 run_backtest.py
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
#  数据结构
# ============================================================
class SignalType(Enum):
    """信号类型"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    # 目标仓位比例（0~1），None 表示全仓买卖
    target_pct: Optional[float] = None
    # 固定股数，target_pct 和 quantity 二选一
    quantity: Optional[int] = None
    reason: str = ""


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int = 0
    avg_cost: float = 0.0          # 持仓均价
    # T+1 锁定：记录今日买入的股数，今日不可卖
    locked_shares: int = 0

    @property
    def sellable_shares(self) -> int:
        """当前可卖股数（排除 T+1 锁定）"""
        return self.shares - self.locked_shares

    def update_on_buy(self, shares: int, price: float):
        """买入后更新持仓"""
        total_cost = self.avg_cost * self.shares + price * shares
        self.shares += shares
        self.avg_cost = total_cost / self.shares if self.shares > 0 else 0.0
        # 新买入股数当日锁定
        self.locked_shares += shares

    def update_on_sell(self, shares: int):
        """卖出后更新持仓"""
        self.shares -= shares
        if self.shares <= 0:
            self.shares = 0
            self.avg_cost = 0.0
        # 卖出时优先消耗锁定股数之外的部分（T+1 已保证 locked 不可卖）

    def reset_day_lock(self):
        """每日开盘时解除昨日锁定"""
        self.locked_shares = 0


@dataclass
class Trade:
    """成交记录"""
    date: pd.Timestamp
    symbol: str
    direction: str        # "BUY" / "SELL"
    shares: int
    price: float
    amount: float         # 成交金额
    commission: float     # 佣金
    tax: float            # 印花税（卖出）
    transfer_fee: float   # 过户费
    total_cost: float     # 买入=金额+费用；卖出=金额-费用


@dataclass
class AShareCostModel:
    """
    A 股交易成本模型（2023 年后标准）
    可按需调整参数适配港美股
    """
    commission_rate: float = 0.00025   # 佣金费率 0.025%
    commission_min: float = 5.0        # 最低佣金 5 元
    stamp_duty_rate: float = 0.0005    # 印花税 0.05%（仅卖出）
    transfer_fee_rate: float = 0.00001 # 过户费 0.001%（沪市双向）

    def calc_buy_cost(self, shares: int, price: float) -> tuple:
        """计算买入成本，返回 (佣金, 印花税, 过户费)"""
        amount = shares * price
        commission = max(amount * self.commission_rate, self.commission_min)
        tax = 0.0  # 买入无印花税
        transfer_fee = amount * self.transfer_fee_rate
        return commission, tax, transfer_fee

    def calc_sell_cost(self, shares: int, price: float) -> tuple:
        """计算卖出成本，返回 (佣金, 印花税, 过户费)"""
        amount = shares * price
        commission = max(amount * self.commission_rate, self.commission_min)
        tax = amount * self.stamp_duty_rate
        transfer_fee = amount * self.transfer_fee_rate
        return commission, tax, transfer_fee


# ============================================================
#  策略基类
# ============================================================
class Strategy(ABC):
    """
    策略抽象基类 —— 继承并实现 on_bar() 即可

    生命周期：
      on_start()  → 回测开始前调用一次（初始化指标等）
      on_bar()    → 每个 bar 调用，返回信号列表
      on_end()    → 回测结束后调用一次
    """
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.engine: Optional[BacktestEngine] = None

    def on_start(self):
        """回测开始前的初始化"""
        pass

    @abstractmethod
    def on_bar(self, date: pd.Timestamp, bar: dict) -> list[Signal]:
        """
        每个 bar 调用一次

        参数:
            date: 当前日期
            bar:  {"symbol": {"open/close/high/low/volume": ...}, ...}
                  包含当前 bar 及之前所有历史数据（通过 self.engine.get_history 访问）

        返回:
            信号列表 [Signal(...), ...]
        """
        ...

    def on_end(self):
        """回测结束"""
        pass

    # ---- 便捷方法 ----
    def get_history(self, symbol: str, field: str = "close", n: int = 60) -> pd.Series:
        """获取最近 n 根 bar 的历史数据"""
        return self.engine.get_history(symbol, field, n)


# ============================================================
#  回测引擎
# ============================================================
class BacktestEngine:
    """
    逐 bar 事件驱动回测引擎

    执行逻辑（杜绝前视偏差）:
      1. bar 收盘 → 策略 on_bar() 生成信号
      2. 下一 bar 开盘 → 按开盘价执行信号
      3. T+1：当日买入的股数当日不可卖出
    """
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        cost_model: AShareCostModel = None,
        slippage: float = 0.0,
        lot_size: int = 100,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.cost_model = cost_model or AShareCostModel()
        self.slippage = slippage          # 滑点（百分比，如 0.001 = 0.1%）
        self.lot_size = lot_size          # 最小交易手数对应的股数

        # 持仓 {symbol: Position}
        self.positions: dict[str, Position] = {}
        # 交易记录
        self.trades: list[Trade] = []
        # 资金曲线
        self.equity_curve: list[dict] = []

        # 数据与状态
        self._data: dict[str, pd.DataFrame] = {}
        self._current_idx: int = 0
        self._dates: list[pd.Timestamp] = []
        self._pending_signals: list[Signal] = []
        self._strategy: Optional[Strategy] = None

    # ---- 数据加载 ----
    def load_data(self, symbol: str, df: pd.DataFrame):
        """
        加载单标的行情数据

        df 需包含列: date(或 index), open, high, low, close, volume
        会自动按日期升序排列
        """
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        df = df.sort_index()
        # 统一列名
        col_map = {c.lower(): c for c in df.columns}
        for std in ["open", "high", "low", "close", "volume"]:
            if std not in df.columns and std.title() in df.columns:
                df = df.rename(columns={std.title(): std})
        self._data[symbol] = df[["open", "high", "low", "close", "volume"]]

    def _build_date_index(self):
        """构建统一的时间轴（取所有标的日期的并集）"""
        all_dates = set()
        for df in self._data.values():
            all_dates.update(df.index)
        self._dates = sorted(all_dates)

    def _get_bar(self, symbol: str, date: pd.Timestamp) -> Optional[dict]:
        """获取某标的在某日的 bar 数据"""
        df = self._data.get(symbol)
        if df is None or date not in df.index:
            return None
        row = df.loc[date]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    # ---- 历史数据访问 ----
    def get_history(self, symbol: str, field: str = "close", n: int = 60) -> pd.Series:
        """获取当前日期之前 n 根 bar 的指定字段"""
        df = self._data.get(symbol)
        if df is None:
            return pd.Series(dtype=float)
        current_date = self._dates[self._current_idx]
        mask = df.index <= current_date
        series = df.loc[mask, field].iloc[-n:]
        return series

    # ---- 仓位查询 ----
    def get_position(self, symbol: str) -> Position:
        """获取持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def get_portfolio_value(self, date: pd.Timestamp) -> float:
        """计算当前总权益 = 现金 + 持仓市值"""
        total = self.cash
        for symbol, pos in self.positions.items():
            if pos.shares > 0:
                bar = self._get_bar(symbol, date)
                if bar:
                    total += pos.shares * bar["close"]
                else:
                    # 无行情时用最新可用收盘价
                    hist = self.get_history(symbol, "close", 1)
                    if len(hist) > 0:
                        total += pos.shares * hist.iloc[-1]
        return total

    # ---- 交易执行 ----
    def _round_to_lot(self, shares: int) -> int:
        """按手数取整（A 股 100 股一手）"""
        return (shares // self.lot_size) * self.lot_size

    def _execute_buy(self, signal: Signal, date: pd.Timestamp):
        """执行买入"""
        symbol = signal.symbol
        bar = self._get_bar(symbol, date)
        if bar is None:
            return

        price = bar["open"] * (1 + self.slippage)  # 滑点

        # 确定买入股数
        if signal.quantity is not None:
            shares = self._round_to_lot(signal.quantity)
        elif signal.target_pct is not None:
            # 目标仓位比例
            portfolio_value = self.get_portfolio_value(date)
            target_value = portfolio_value * signal.target_pct
            current_pos = self.get_position(symbol)
            current_value = current_pos.shares * price
            buy_value = target_value - current_value
            if buy_value > 0:
                shares = self._round_to_lot(int(buy_value / price))
            else:
                shares = 0
        else:
            # 默认全仓买入
            shares = self._round_to_lot(int(self.cash * 0.95 / price))

        if shares < self.lot_size:
            return  # 不足一手

        amount = shares * price
        commission, tax, transfer_fee = self.cost_model.calc_buy_cost(shares, price)
        total_cost = amount + commission + transfer_fee

        if total_cost > self.cash:
            # 资金不足，调整股数
            max_shares = self._round_to_lot(
                int(self.cash / (price * (1 + self.cost_model.commission_rate)))
            )
            if max_shares < self.lot_size:
                return
            shares = max_shares
            amount = shares * price
            commission, tax, transfer_fee = self.cost_model.calc_buy_cost(shares, price)
            total_cost = amount + commission + transfer_fee

        # 执行
        self.cash -= total_cost
        pos = self.get_position(symbol)
        pos.update_on_buy(shares, price)

        self.trades.append(Trade(
            date=date, symbol=symbol, direction="BUY",
            shares=shares, price=price, amount=amount,
            commission=commission, tax=tax, transfer_fee=transfer_fee,
            total_cost=total_cost,
        ))

    def _execute_sell(self, signal: Signal, date: pd.Timestamp):
        """执行卖出"""
        symbol = signal.symbol
        bar = self._get_bar(symbol, date)
        if bar is None:
            return

        pos = self.get_position(symbol)
        if pos.sellable_shares <= 0:
            return  # T+1 限制：无可卖股数

        price = bar["open"] * (1 - self.slippage)

        # 确定卖出股数
        if signal.quantity is not None:
            shares = min(self._round_to_lot(signal.quantity), pos.sellable_shares)
        elif signal.target_pct is not None and signal.target_pct == 0:
            # 清仓
            shares = pos.sellable_shares
        else:
            # 默认全部可卖
            shares = pos.sellable_shares

        if shares < self.lot_size and shares < pos.sellable_shares:
            shares = self._round_to_lot(shares)
        if shares <= 0:
            return

        amount = shares * price
        commission, tax, transfer_fee = self.cost_model.calc_sell_cost(shares, price)
        net_proceeds = amount - commission - tax - transfer_fee

        # 执行
        self.cash += net_proceeds
        pos.update_on_sell(shares)

        self.trades.append(Trade(
            date=date, symbol=symbol, direction="SELL",
            shares=shares, price=price, amount=amount,
            commission=commission, tax=tax, transfer_fee=transfer_fee,
            total_cost=net_proceeds,
        ))

    def _execute_pending_signals(self, date: pd.Timestamp):
        """在开盘时执行上一 bar 生成的待处理信号"""
        for signal in self._pending_signals:
            if signal.signal_type == SignalType.BUY:
                self._execute_buy(signal, date)
            elif signal.signal_type == SignalType.SELL:
                self._execute_sell(signal, date)
        self._pending_signals = []

    # ---- 主循环 ----
    def run(self, strategy: Strategy) -> dict:
        """
        运行回测

        返回:
            {
                "equity_curve": pd.DataFrame,
                "trades": pd.DataFrame,
                "final_equity": float,
                "total_return": float,
            }
        """
        self._strategy = strategy
        strategy.engine = self
        self._build_date_index()

        if len(self._dates) == 0:
            raise ValueError("无数据，请先 load_data()")

        strategy.on_start()

        for i in range(len(self._dates)):
            self._current_idx = i
            date = self._dates[i]

            # 每日开盘：解除 T+1 锁定，执行待处理信号
            for pos in self.positions.values():
                pos.reset_day_lock()
            self._execute_pending_signals(date)

            # 收盘：收集当前 bar 数据，调用策略
            bars = {}
            for symbol in self._data:
                bar = self._get_bar(symbol, date)
                if bar:
                    bars[symbol] = bar

            if bars:
                signals = strategy.on_bar(date, bars)
                if signals:
                    self._pending_signals = signals if isinstance(signals, list) else [signals]

            # 记录当日权益
            equity = self.get_portfolio_value(date)
            self.equity_curve.append({
                "date": date,
                "equity": equity,
                "cash": self.cash,
                "position_value": equity - self.cash,
            })

        strategy.on_end()

        return self._build_results()

    def _build_results(self) -> dict:
        """整理回测结果"""
        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        trades_df = pd.DataFrame([
            {
                "date": t.date, "symbol": t.symbol, "direction": t.direction,
                "shares": t.shares, "price": round(t.price, 4),
                "amount": round(t.amount, 2), "commission": round(t.commission, 2),
                "tax": round(t.tax, 2), "transfer_fee": round(t.transfer_fee, 4),
                "net": round(t.total_cost, 2),
            }
            for t in self.trades
        ])
        if not trades_df.empty:
            trades_df = trades_df.set_index("date")

        final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        return {
            "equity_curve": equity_df,
            "trades": trades_df,
            "final_equity": final_equity,
            "total_return": total_return,
        }


# ============================================================
#  便捷函数：快速回测单标的
# ============================================================
def quick_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    symbol: str = "TEST",
    initial_capital: float = 1_000_000,
    **kwargs,
) -> dict:
    """
    快速回测单标的

    df: 含 open/high/low/close/volume 的 DataFrame
    strategy: 策略实例
    """
    engine = BacktestEngine(initial_capital=initial_capital, **kwargs)
    engine.load_data(symbol, df)
    return engine.run(strategy)
