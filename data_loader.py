"""
data_loader.py — 行情数据加载器

数据源优先级：
  1. westock-data CLI（A 股前复权日线，最权威）
  2. yfinance（美股/港股，需联网）
  3. 合成数据（几何布朗运动，离线可用，仅用于框架验证）

用法：
    df = load_a_share("600519.SH", days=500)
    df = load_synthetic("DEMO", days=500, seed=42)
"""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _try_westock(code: str, days: int) -> pd.DataFrame | None:
    """尝试通过 westock-data CLI 获取 A 股数据"""
    if not shutil.which("westock-data"):
        return None
    try:
        cmd = ["westock-data", "kline", code, "--period", "day",
               "--limit", str(days), "--fq", "qfq"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        # westock-data 输出 Markdown 表格，需解析
        return _parse_markdown_table(result.stdout)
    except Exception:
        return None


def _parse_markdown_table(md: str) -> pd.DataFrame | None:
    """解析 westock-data 的 Markdown 表格输出为 DataFrame"""
    lines = [l.strip() for l in md.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return None
    # 找到表头行
    header_idx = None
    for i, line in enumerate(lines):
        if "|" in line and "日期" in line:
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(lines):
            if "|" in line and "date" in line.lower():
                header_idx = i
                break
    if header_idx is None:
        return None

    headers = [h.strip() for h in lines[header_idx].split("|") if h.strip()]
    rows = []
    for line in lines[header_idx + 2:]:  # 跳过分隔行
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) == len(headers):
            rows.append(cells)

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=headers)
    # 统一列名
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "日期" in c or cl == "date" or "时间" in c:
            col_map[c] = "date"
        elif "开" in c or cl == "open":
            col_map[c] = "open"
        elif "高" in c or cl == "high":
            col_map[c] = "high"
        elif "低" in c or cl == "low":
            col_map[c] = "low"
        elif "收" in c or cl == "close":
            col_map[c] = "close"
        elif "量" in c or cl == "volume":
            col_map[c] = "volume"
    df = df.rename(columns=col_map)
    needed = ["date", "open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df if len(df) > 0 else None


def _try_yfinance(symbol: str, days: int) -> pd.DataFrame | None:
    """尝试通过 yfinance 获取数据"""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        period = f"{days}d"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception:
        return None


def load_synthetic(symbol: str = "DEMO", days: int = 500, seed: int = 42,
                   start_price: float = 50.0) -> pd.DataFrame:
    """
    生成合成行情数据（几何布朗运动 + 波动率聚集）

    用于框架验证和离线演示，非真实行情
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    # GBM 参数
    annual_drift = 0.08          # 年化漂移
    annual_vol = 0.25            # 年化波动
    dt = 1 / 252
    mu = annual_drift * dt
    sigma = annual_vol * np.sqrt(dt)

    # GARCH-like 波动率聚集
    base_vol = sigma
    vol_series = np.zeros(days)
    vol_series[0] = base_vol
    for i in range(1, days):
        vol_series[i] = 0.94 * vol_series[i-1] + 0.06 * base_vol + rng.normal(0, 0.002)

    # 价格路径
    log_returns = rng.normal(mu, vol_series)
    prices = np.zeros(days)
    prices[0] = start_price
    for i in range(1, days):
        prices[i] = prices[i-1] * np.exp(log_returns[i])

    # 生成 OHLC
    opens = np.roll(prices, 1)
    opens[0] = start_price
    intraday_range = prices * rng.uniform(0.005, 0.02, days)
    highs = np.maximum(opens, prices) + rng.uniform(0, intraday_range)
    lows = np.minimum(opens, prices) - rng.uniform(0, intraday_range)
    volumes = rng.integers(500_000, 5_000_000, days).astype(float)

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(prices, 2),
        "volume": volumes,
    })
    return df


def load_a_share(code: str, days: int = 500) -> pd.DataFrame:
    """
    加载 A 股数据，自动选择数据源

    code: 如 "600519.SH"、"000001.SZ"
    days: 获取最近多少个交易日
    """
    # 优先 westock-data
    df = _try_westock(code, days)
    if df is not None:
        print(f"[数据] westock-data 加载 {code}: {len(df)} 条")
        return df

    # 降级 yfinance
    yf_code = code.replace(".SH", ".SS").replace(".SZ", ".SZ")
    df = _try_yfinance(yf_code, days)
    if df is not None:
        print(f"[数据] yfinance 加载 {code}: {len(df)} 条")
        return df

    # 最终降级合成数据
    print(f"[数据] {code} 无法获取真实数据，使用合成数据（仅用于框架验证）")
    return load_synthetic(code, days)


if __name__ == "__main__":
    # 快速测试
    df = load_synthetic("TEST", 100, seed=42)
    print(df.head())
    print(f"\n共 {len(df)} 条记录")
