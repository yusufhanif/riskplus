"""Portfolio weight parsing, normalization, and validation helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def detect_weight_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Detect likely fund and weight columns from a weight table."""
    fund_synonyms = ['fund', 'fund name', 'asset', 'asset name', 'manager', 'product']
    weight_synonyms = ['weight', 'allocation', 'weight %', 'portfolio weight', 'current weight']

    def score_columns(synonyms: list[str]) -> list[dict[str, Any]]:
        scores: list[dict[str, Any]] = []
        for column in df.columns:
            normalized = _normalize_text(column)
            score = 0.0
            if normalized in synonyms:
                score = 1.0
            elif any(syn in normalized for syn in synonyms):
                score = 0.9
            else:
                score = max(SequenceMatcher(None, normalized, synonym).ratio() for synonym in synonyms)
            scores.append({'column': column, 'score': float(score)})
        scores.sort(key=lambda item: item['score'], reverse=True)
        return scores

    fund_candidates = score_columns(fund_synonyms)
    weight_candidates = score_columns(weight_synonyms)

    return {
        'fund_column': fund_candidates[0]['column'] if fund_candidates else None,
        'weight_column': weight_candidates[0]['column'] if weight_candidates else None,
        'fund_candidates': fund_candidates,
        'weight_candidates': weight_candidates,
        'fund_synonyms': fund_synonyms,
        'weight_synonyms': weight_synonyms,
    }


def prepare_weights_table(df: pd.DataFrame, fund_col: str, weight_col: str) -> pd.DataFrame:
    """Return a cleaned two-column weights table with canonical names."""
    table = df[[fund_col, weight_col]].copy()
    table.columns = ['Fund', 'Weight']
    table['Fund'] = table['Fund'].astype(str).str.strip()
    table.loc[table['Fund'].isin(['', 'nan', 'None']), 'Fund'] = pd.NA
    table['Weight'] = pd.to_numeric(table['Weight'], errors='coerce')
    return table


def normalize_portfolio_weights(weights: pd.Series) -> pd.Series:
    """Normalize a weight series to sum to 1.0."""
    numeric = pd.to_numeric(weights, errors='coerce').fillna(0.0).astype(float)
    total = float(numeric.sum())
    if total <= 0:
        return numeric
    return numeric / total


def validate_portfolio_weights(weights: pd.Series, asset_cols: list[str]) -> tuple[list[str], list[str]]:
    """Validate a weight series against the selected portfolio columns."""
    errors: list[str] = []
    warnings: list[str] = []

    numeric = pd.to_numeric(weights, errors='coerce')
    if numeric.index.duplicated().any():
        duplicates = sorted(set(numeric.index[numeric.index.duplicated()].tolist()))
        errors.append(f"Duplicate fund names in weights file: {', '.join(duplicates)}")

    if numeric.isna().any():
        warnings.append('Some weights were blank or nonnumeric and were treated as 0%.')

    negative_names = numeric[numeric < 0].index.tolist()
    if negative_names:
        errors.append(f"Negative weights are not allowed: {', '.join(map(str, negative_names))}")

    total = float(numeric.fillna(0.0).sum())
    if total <= 0:
        errors.append('Zero total weight detected.')

    missing_weight_funds = [asset for asset in asset_cols if asset not in numeric.index]
    extra_weight_rows = [name for name in numeric.index.unique() if name not in asset_cols]
    if missing_weight_funds:
        warnings.append(f"{len(missing_weight_funds)} selected funds were missing from the weights file and were assigned 0% weight.")
    if extra_weight_rows:
        warnings.append(f"{len(extra_weight_rows)} rows in the weights file do not match any selected fund return column.")

    zero_weight_funds = numeric.index[(numeric.fillna(0.0) == 0.0)].tolist()
    if zero_weight_funds:
        warnings.append(f"{len(zero_weight_funds)} selected funds have 0% weight.")

    return errors, warnings


def match_weight_names_to_assets(weight_names: list[str], asset_cols: list[str]) -> pd.DataFrame:
    """Match uploaded weight names to selected asset columns."""
    rows: list[dict[str, Any]] = []
    normalized_assets = {asset: _normalize_text(asset) for asset in asset_cols}

    for weight_name in weight_names:
        normalized_weight = _normalize_text(weight_name)
        exact_match = next((asset for asset, normalized_asset in normalized_assets.items() if normalized_asset == normalized_weight), None)
        if exact_match is not None:
            rows.append(
                {
                    'Uploaded Fund Name': weight_name,
                    'Matched Return Column': exact_match,
                    'Match Confidence': 1.0,
                    'User Confirmed': True,
                }
            )
            continue

        scored_assets = [
            (asset, SequenceMatcher(None, normalized_weight, normalized_asset).ratio())
            for asset, normalized_asset in normalized_assets.items()
        ]
        if scored_assets:
            best_asset, best_score = max(scored_assets, key=lambda item: item[1])
        else:
            best_asset, best_score = '', 0.0

        rows.append(
            {
                'Uploaded Fund Name': weight_name,
                'Matched Return Column': best_asset if best_score >= 0.6 else '',
                'Match Confidence': float(best_score),
                'User Confirmed': False,
            }
        )

    return pd.DataFrame(rows)


def build_asset_weight_series(asset_cols: list[str], weight_table: pd.DataFrame, normalize: bool = True) -> pd.Series:
    """Build a decimal weight Series indexed by asset columns."""
    if weight_table.empty:
        return pd.Series(0.0, index=asset_cols, dtype=float)

    fund_col = 'Fund' if 'Fund' in weight_table.columns else weight_table.columns[0]
    weight_col = 'Weight' if 'Weight' in weight_table.columns else weight_table.columns[1]

    weights = weight_table[[fund_col, weight_col]].copy()
    weights[fund_col] = weights[fund_col].astype(str).str.strip()
    weights[weight_col] = pd.to_numeric(weights[weight_col], errors='coerce').fillna(0.0)

    collapsed = weights.groupby(fund_col, dropna=False)[weight_col].sum()
    collapsed = collapsed.reindex(asset_cols).fillna(0.0).astype(float)

    if normalize:
        return normalize_portfolio_weights(collapsed)
    return collapsed