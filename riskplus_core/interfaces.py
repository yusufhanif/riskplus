"""Future-facing interfaces for model sophistication.

These are intentionally lightweight contracts only. They make it easier to
introduce new covariance estimators, simulation engines, factor selectors,
and export backends without changing the current analysis outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(slots=True)
class StressScenario:
    """Named shock definition for future scenario and stress testing."""

    name: str
    description: str = ""
    factor_shocks: dict[str, float] | None = None
    portfolio_shock: float | None = None


@runtime_checkable
class CovarianceEstimator(Protocol):
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate a covariance matrix from return history."""


@runtime_checkable
class SimulationEngine(Protocol):
    def simulate(
        self,
        historical_returns: pd.Series | pd.DataFrame,
        *,
        n_sims: int,
        random_seed: int,
    ) -> pd.DataFrame:
        """Generate future return paths from a chosen simulation model."""


@runtime_checkable
class FactorModelSelector(Protocol):
    def select(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> list[str]:
        """Choose a factor subset for future stepwise AIC/BIC workflows."""


@runtime_checkable
class RollingBetaCalculator(Protocol):
    def calculate(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        *,
        window: int,
    ) -> pd.DataFrame:
        """Compute rolling beta diagnostics."""


@runtime_checkable
class ExportWriter(Protocol):
    def write_excel(self, output_path: str, results: object) -> None:
        """Write a workbook export for the supplied analysis results."""

    def write_pdf(self, output_path: str, results: object) -> None:
        """Write a PDF export for the supplied analysis results."""
