"""
strategies.py — 实战策略集

三个策略均含完整的风控逻辑（止损/止盈/仓位管理），
可作为模板扩展自定义策略。

策略一：DualMA_ATR — 双均线交叉 + ATR 动态止损（趋势跟踪）
策略二：RSI_Reversal — RSI 超卖反弹（均值回归）
策略三：Donchian_Breakout — 唐奇安通道突破（海龟经典）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_engine import BacktestEngine, Signal, SignalType, Strategy
from indicators import (
    SMA, EMA, RSI, ATR, DonchianChannel, crossover, crossunder,
)


# ============================================================
#  策略一：双均线交叉 + ATR 止损（趋势跟踪）
# ============================================================
class DualMA_ATR_Strategy(Strategy):
    """
    逻辑：
      - 快线上穿慢线 → 买入（金叉）
      - 快线下穿慢线 → 卖出（死叉）
      - 持仓期间用 ATR 倍数做动态止损（跟踪止损）
      - 仓位 = 风险百分比 / (ATR倍数 × ATR单价)

    适用：趋势行情（如牛市主升浪）
    不适用：震荡行情（频繁假信号）
    """

    def __init__(
        self,
        symbol: str,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_period: int = 20,
        atr_multiplier: float = 2.5,     # 止损 = 2.5 × ATR
        risk_pct: float = 0.02,          # 每笔风险 2% 总资金
        warmup: int = 60,
    ):
        super().__init__(name=f"DualMA_ATR({fast_period}/{slow_period})")
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_pct = risk_pct
        self.warmup = warmup

        # 运行时状态
        self._stop_loss: float = 0.0
        self._entry_price: float = 0.0
        self._bar_count: int = 0

    def on_start(self):
        self._bar_count = 0

    def on_bar(self, date: pd.Timestamp, bars: dict) -> list[Signal]:
        self._bar_count += 1
        if self.symbol not in bars or self._bar_count < self.warmup:
            return []

        bar = bars[self.symbol]
        signals = []

        # 计算指标
        close = self.get_history(self.symbol, "close", max(self.slow_period, self.atr_period) + 5)
        high = self.get_history(self.symbol, "high", self.atr_period + 5)
        low = self.get_history(self.symbol, "low", self.atr_period + 5)

        if len(close) < self.slow_period + 2:
            return []

        fast_ma = SMA(close, self.fast_period)
        slow_ma = SMA(close, self.slow_period)
        atr = ATR(high, low, close, self.atr_period)

        if len(fast_ma) < 2 or pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]):
            return []

        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

        pos = self.engine.get_position(self.symbol)

        # ---- 持仓中：检查止损/死叉 ----
        if pos.shares > 0:
            # 动态跟踪止损：价格上涨则上移止损
            new_stop = bar["close"] - self.atr_multiplier * current_atr
            if new_stop > self._stop_loss:
                self._stop_loss = new_stop

            # 触发止损
            if bar["close"] <= self._stop_loss:
                signals.append(Signal(
                    symbol=self.symbol, signal_type=SignalType.SELL,
                    target_pct=0, reason=f"ATR止损@stop={self._stop_loss:.2f}",
                ))
                self._stop_loss = 0.0
                return signals

            # 死叉卖出
            if prev_fast >= prev_slow and current_fast < current_slow:
                signals.append(Signal(
                    symbol=self.symbol, signal_type=SignalType.SELL,
                    target_pct=0, reason="死叉",
                ))
                self._stop_loss = 0.0
                return signals

        # ---- 空仓：检查金叉买入 ----
        else:
            if prev_fast <= prev_slow and current_fast > current_slow:
                # 按风险百分比计算仓位
                stop_distance = self.atr_multiplier * current_atr
                if stop_distance > 0:
                    portfolio_value = self.engine.get_portfolio_value(date)
                    risk_amount = portfolio_value * self.risk_pct
                    shares = int(risk_amount / stop_distance / self.engine.lot_size) * self.engine.lot_size
                    if shares >= self.engine.lot_size:
                        target_pct = (shares * bar["close"]) / portfolio_value
                        target_pct = min(target_pct, 0.95)  # 不超过 95%
                        signals.append(Signal(
                            symbol=self.symbol, signal_type=SignalType.BUY,
                            target_pct=target_pct, reason="金叉+ATR仓位",
                        ))
                        self._entry_price = bar["close"]
                        self._stop_loss = bar["close"] - stop_distance

        return signals


# ============================================================
#  策略二：RSI 超卖反弹（均值回归）
# ============================================================
class RSI_Reversal_Strategy(Strategy):
    """
    逻辑：
      - RSI < 超卖阈值 → 买入（超跌反弹）
      - RSI > 超买阈值 → 卖出（获利了结）
      - 固定仓位（如 30%），分批建仓可选

    适用：震荡行情
    不适用：单边趋势（超卖可以更超卖）
    """

    def __init__(
        self,
        symbol: str,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 65,
        position_pct: float = 0.30,    # 每次建仓 30%
        max_position_pct: float = 0.90,
        stop_loss_pct: float = 0.08,   # 固定止损 8%
        warmup: int = 30,
    ):
        super().__init__(name=f"RSI_Reversal({rsi_period},{oversold}/{overbought})")
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.position_pct = position_pct
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.warmup = warmup
        self._bar_count = 0
        self._entry_price = 0.0

    def on_start(self):
        self._bar_count = 0

    def on_bar(self, date: pd.Timestamp, bars: dict) -> list[Signal]:
        self._bar_count += 1
        if self.symbol not in bars or self._bar_count < self.warmup:
            return []

        bar = bars[self.symbol]
        signals = []

        close = self.get_history(self.symbol, "close", self.rsi_period + 10)
        if len(close) < self.rsi_period + 2:
            return []

        rsi = RSI(close, self.rsi_period)
        if pd.isna(rsi.iloc[-1]):
            return []

        current_rsi = rsi.iloc[-1]
        pos = self.engine.get_position(self.symbol)

        # ---- 持仓中 ----
        if pos.shares > 0:
            # 固定止损
            if self._entry_price > 0:
                loss_pct = (self._entry_price - bar["close"]) / self._entry_price
                if loss_pct >= self.stop_loss_pct:
                    signals.append(Signal(
                        symbol=self.symbol, signal_type=SignalType.SELL,
                        target_pct=0, reason=f"止损{loss_pct*100:.1f}%",
                    ))
                    self._entry_price = 0.0
                    return signals

            # RSI 超买卖出
            if current_rsi >= self.overbought:
                signals.append(Signal(
                    symbol=self.symbol, signal_type=SignalType.SELL,
                    target_pct=0, reason=f"RSI超买={current_rsi:.1f}",
                ))
                self._entry_price = 0.0
                return signals

        # ---- 空仓或仓位不足：RSI 超卖买入 ----
        else:
            if current_rsi <= self.oversold:
                portfolio_value = self.engine.get_portfolio_value(date)
                current_pos_value = pos.shares * bar["close"]
                if current_pos_value / portfolio_value < self.max_position_pct:
                    signals.append(Signal(
                        symbol=self.symbol, signal_type=SignalType.BUY,
                        target_pct=self.position_pct, reason=f"RSI超卖={current_rsi:.1f}",
                    ))
                    self._entry_price = bar["close"]

        return signals


# ============================================================
#  策略三：唐奇安通道突破（海龟经典）
# ============================================================
class Donchian_Breakout_Strategy(Strategy):
    """
    逻辑：
      - 收盘价突破过去 N 日最高价 → 买入
      - 收盘价跌破过去 M 日最低价 → 卖出（M 通常 < N）
      - 仓位基于 ATR 风险控制

    经典海龟策略简化版，适合趋势行情
    """

    def __init__(
        self,
        symbol: str,
        entry_period: int = 20,       # 突破买入周期
        exit_period: int = 10,        # 突破卖出周期
        atr_period: int = 20,
        risk_pct: float = 0.01,       # 每笔风险 1%
        atr_stop_mult: float = 2.0,   # 止损 ATR 倍数
        warmup: int = 40,
    ):
        super().__init__(name=f"Donchian({entry_period}/{exit_period})")
        self.symbol = symbol
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_pct = risk_pct
        self.atr_stop_mult = atr_stop_mult
        self.warmup = warmup
        self._bar_count = 0
        self._stop_loss = 0.0

    def on_start(self):
        self._bar_count = 0

    def on_bar(self, date: pd.Timestamp, bars: dict) -> list[Signal]:
        self._bar_count += 1
        if self.symbol not in bars or self._bar_count < self.warmup:
            return []

        bar = bars[self.symbol]
        signals = []

        n = max(self.entry_period, self.exit_period, self.atr_period) + 5
        close = self.get_history(self.symbol, "close", n)
        high = self.get_history(self.symbol, "high", n)
        low = self.get_history(self.symbol, "low", n)

        if len(close) < self.entry_period + 2:
            return []

        # 唐奇安通道（shift 避免前视）
        entry_upper = high.rolling(self.entry_period).max().shift(1)
        exit_lower = low.rolling(self.exit_period).min().shift(1)
        atr = ATR(high, low, close, self.atr_period)

        if pd.isna(entry_upper.iloc[-1]) or pd.isna(exit_lower.iloc[-1]):
            return []

        upper_line = entry_upper.iloc[-1]
        lower_line = exit_lower.iloc[-1]
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

        pos = self.engine.get_position(self.symbol)

        # ---- 持仓中：止损或跌破退出线 ----
        if pos.shares > 0:
            # ATR 止损
            if bar["close"] <= self._stop_loss and self._stop_loss > 0:
                signals.append(Signal(
                    symbol=self.symbol, signal_type=SignalType.SELL,
                    target_pct=0, reason=f"ATR止损@stop={self._stop_loss:.2f}",
                ))
                self._stop_loss = 0.0
                return signals

            # 跌破退出线
            if bar["close"] <= lower_line:
                signals.append(Signal(
                    symbol=self.symbol, signal_type=SignalType.SELL,
                    target_pct=0, reason=f"跌破{self.exit_period}日低点",
                ))
                self._stop_loss = 0.0
                return signals

        # ---- 空仓：突破买入 ----
        else:
            if bar["close"] > upper_line and current_atr > 0:
                stop_distance = self.atr_stop_mult * current_atr
                portfolio_value = self.engine.get_portfolio_value(date)
                risk_amount = portfolio_value * self.risk_pct
                shares = int(risk_amount / stop_distance / self.engine.lot_size) * self.engine.lot_size
                if shares >= self.engine.lot_size:
                    target_pct = min((shares * bar["close"]) / portfolio_value, 0.95)
                    signals.append(Signal(
                        symbol=self.symbol, signal_type=SignalType.BUY,
                        target_pct=target_pct, reason=f"突破{self.entry_period}日高点",
                    ))
                    self._stop_loss = bar["close"] - stop_distance

        return signals
