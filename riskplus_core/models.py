"""Dataclasses for structured inputs and outputs used by the backend pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ColumnMapping:
    date_col: str
    fund_cols: list[str] = field(default_factory=list)
    factor_cols: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PortfolioMetadata:
    report_name: str = "Portfolio Risk Analysis"
    portfolio_value: float = 1_000_000.0
    rf_rate: float = 0.02


@dataclass(slots=True)
class RiskSettings:
    confidence: float = 0.95
    num_sims: int = 50_000
    corr_method: str = "Classical"
    ewma_decay: float = 0.94
    dist_type: str = "Student-t"


@dataclass(slots=True)
class AnalysisWindow:
    start_date: Any | None = None
    end_date: Any | None = None


@dataclass(slots=True)
class PreparedData:
    raw: pd.DataFrame | None = None
    merged: pd.DataFrame | None = None
    portfolio: pd.Series | None = None
    factors: pd.DataFrame | None = None
    fund_weights: pd.Series | None = None


@dataclass(slots=True)
class CoreAnalysisResults:
    """Bundle the outputs returned by the reusable core analysis engine."""
    frequency: str = "unknown"
    portfolio: pd.Series | None = None
    asset_weights: pd.Series | None = None
    factors: pd.DataFrame | None = None
    simulated_returns: pd.DataFrame | None = None
    hist_stats: dict[str, float] = field(default_factory=dict)
    sim_stats: dict[str, float] = field(default_factory=dict)
    ols_results: dict[str, Any] = field(default_factory=dict)
    mc_contribs: dict[str, Any] = field(default_factory=dict)
    sys_spec: dict[str, Any] = field(default_factory=dict)
    factor_contrib: dict[str, Any] = field(default_factory=dict)
    rb_etl: pd.DataFrame | None = None
    rb_stdev: pd.DataFrame | None = None
    bucket_exposures: pd.DataFrame | None = None