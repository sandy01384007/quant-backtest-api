"""
indicators.py — 常用技术指标库

全部基于 pandas 向量化实现，无外部依赖
所有函数接受 pd.Series（价格序列），返回 pd.Series（指标序列）
"""
import numpy as np
import pandas as pd


# ============================================================
#  均线类
# ============================================================
def SMA(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=period, min_periods=period).mean()


def EMA(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=period, adjust=False).mean()


def WMA(series: pd.Series, period: int) -> pd.Series:
    """加权移动平均（越近权重越大）"""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# ============================================================
#  动量类
# ============================================================
def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标 RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder 平滑法
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def MACD(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 指标
    返回: (macd_line, signal_line, histogram)
    """
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def ROC(series: pd.Series, period: int = 12) -> pd.Series:
    """变动率指标 ROC"""
    return (series / series.shift(period) - 1) * 100


# ============================================================
#  波动率类
# ============================================================
def ATR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    平均真实波幅 ATR

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = TR 的 Wilder 平滑
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def BollingerBands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林带
    返回: (upper, middle, lower)
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


# ============================================================
#  通道类
# ============================================================
def DonchianChannel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    唐奇安通道
    返回: (upper, middle, lower)
    upper = 过去 period 日最高价（不含当日，避免前视）
    lower = 过去 period 日最低价（不含当日）
    """
    upper = high.rolling(window=period).max().shift(1)
    lower = low.rolling(window=period).min().shift(1)
    middle = (upper + lower) / 2
    return upper, middle, lower


def KeltnerChannel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    肯特纳通道（基于 ATR）
    返回: (upper, middle, lower)
    """
    middle = EMA(close, period)
    atr = ATR(high, low, close, period)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return upper, middle, lower


# ============================================================
#  成交量类
# ============================================================
def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮指标 OBV"""
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv


def VWAP(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """成交量加权平均价（滚动）"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()


# ============================================================
#  辅助函数
# ============================================================
def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """金叉判断：series1 上穿 series2"""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """死叉判断：series1 下穿 series2"""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
