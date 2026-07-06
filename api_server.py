"""
api_server.py — 量化策略回测 REST API (FastAPI)

端点：
  POST /backtest       — 运行单策略回测
  POST /backtest/all   — 运行所有策略回测
  GET  /health         — 健康检查
  GET  /docs           — API 文档 (Swagger UI)

部署：
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backtest_engine import BacktestEngine
from strategies import (
    DualMA_ATR_Strategy,
    RSI_Reversal_Strategy,
    Donchian_Breakout_Strategy,
)
from data_loader import load_a_share, load_synthetic
from performance import calc_metrics

# ---------- FastAPI app ----------

app = FastAPI(
    title="量化策略回测引擎 API",
    description="A 股/港股/美股多市场量化策略回测服务。支持规则型策略开发、事件研究、多标的选股与组合再平衡。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 全局状态 ----------

_tasks: dict[str, dict] = {}  # task_id -> status info
_lock = threading.Lock()

STRATEGY_MAP = {
    "dual_ma": DualMA_ATR_Strategy,
    "rsi": RSI_Reversal_Strategy,
    "donchian": Donchian_Breakout_Strategy,
}

STRATEGY_LABELS = {
    "dual_ma": "均线交叉 + ATR 过滤",
    "rsi": "RSI 超买超卖反转",
    "donchian": "唐奇安通道突破",
}


# ---------- Pydantic 模型 ----------


class BacktestRequest(BaseModel):
    strategy: str = Field(
        "all",
        pattern=r"^(all|dual_ma|rsi|donchian)$",
        description="策略类型: all / dual_ma / rsi / donchian",
    )
    symbol: str = Field("DEMO", description="标的代码 (如 600519.SH, 0700.HK, AAPL, DEMO=合成数据)")
    days: int = Field(500, ge=30, le=2000, description="回测数据天数 (30~2000)")
    capital: float = Field(1_000_000, ge=10_000, le=100_000_000, description="初始资金")
    fast_mode: bool = Field(
        True, description="快速模式: True=仅返回指标摘要, False=返回完整结果(含CSV)"
    )


class BacktestResponse(BaseModel):
    task_id: str
    status: str
    message: str


# ---------- 辅助函数 ----------


def _run_backtest(task_id: str, req: BacktestRequest):
    """在后台线程中运行回测"""
    try:
        with _lock:
            _tasks[task_id] = {"status": "running", "progress": "加载数据中..."}

        # ---- 加载数据 ----
        if req.symbol == "DEMO":
            df = load_synthetic("DEMO", req.days, seed=42)
            data_source = "合成数据(GBM)"
        else:
            df = load_a_share(req.symbol, req.days)
            data_source = f"westock-data/{req.symbol}"

        # ---- 选择策略 ----
        if req.strategy == "all":
            strategies = [
                ("dual_ma", DualMA_ATR_Strategy, {}),
                ("rsi", RSI_Reversal_Strategy, {}),
                ("donchian", Donchian_Breakout_Strategy, {}),
            ]
        else:
            cls = STRATEGY_MAP[req.strategy]
            strategies = [(req.strategy, cls, {})]

        results = []

        for key, strat_cls, kwargs in strategies:
            with _lock:
                _tasks[task_id]["progress"] = f"运行策略: {STRATEGY_LABELS.get(key, strat_cls.__name__)}..."

            engine = BacktestEngine(initial_capital=req.capital)
            engine.load_data(req.symbol, df)
            strategy = strat_cls(symbol=req.symbol, **kwargs)
            engine_results = engine.run(strategy)
            metrics = calc_metrics(
                engine_results["equity_curve"],
                engine_results["trades"],
                req.capital,
            )

            # 构建结果
            result_item = {
                "strategy": key,
                "strategy_label": STRATEGY_LABELS.get(key, strat_cls.__name__),
                "metrics": {
                    "initial_capital": metrics["initial_capital"],
                    "final_equity": metrics["final_equity"],
                    "total_return_pct": round(metrics["total_return"], 4),
                    "annual_return_pct": round(metrics["annual_return"], 4),
                    "annual_std_pct": round(metrics["annual_std"], 4),
                    "sharpe": round(metrics["sharpe"], 4),
                    "max_drawdown_pct": round(metrics["max_drawdown"], 4),
                    "total_trades": metrics["total_trades"],
                    "win_rate_pct": round(metrics["win_rate"], 2),
                    "profit_factor": round(metrics["profit_factor"], 4),
                    "calmar": round(metrics["calmar"], 4),
                },
            }

            if not req.fast_mode:
                # 完整模式：含 CSV 数据
                result_item["equity_curve"] = (
                    engine_results["equity_curve"].to_dict(orient="records")
                )
                result_item["trades"] = (
                    engine_results["trades"].to_dict(orient="records") if not engine_results["trades"].empty else []
                )

            results.append(result_item)

        with _lock:
            _tasks[task_id] = {
                "status": "completed",
                "progress": "回测完成",
                "data": {
                    "symbol": req.symbol,
                    "data_source": data_source,
                    "days": len(df),
                    "date_range": {
                        "start": str(df["date"].iloc[0].date()),
                        "end": str(df["date"].iloc[-1].date()),
                    },
                    "initial_capital": req.capital,
                    "strategy_count": len(results),
                    "results": results,
                },
            }

    except Exception as e:
        with _lock:
            _tasks[task_id] = {
                "status": "failed",
                "progress": "回测失败",
                "error": str(e),
            }


# ---------- API 端点 ----------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "量化策略回测引擎",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/backtest", response_model=BacktestResponse)
async def start_backtest(req: BacktestRequest):
    """启动一个回测任务，返回 task_id 用于轮询结果"""
    task_id = str(uuid.uuid4())[:8]
    thread = threading.Thread(target=_run_backtest, args=(task_id, req), daemon=True)
    thread.start()
    return BacktestResponse(
        task_id=task_id,
        status="accepted",
        message=f"回测任务已提交 (task_id={task_id})，使用 GET /backtest/{task_id} 查询结果",
    )


@app.get("/backtest/{task_id}")
async def get_backtest_result(task_id: str):
    """查询回测任务状态和结果"""
    with _lock:
        task = _tasks.get(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", ""),
    }

    if task["status"] == "completed":
        response["data"] = task["data"]
    elif task["status"] == "failed":
        response["error"] = task.get("error", "未知错误")

    return response


# ---------- 直接入口 ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
