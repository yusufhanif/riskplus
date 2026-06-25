from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from .data import annualization_factor

DEFAULT_CONFIDENCE = 0.95
DEFAULT_NUM_SIMULATIONS = 50000
DEFAULT_RF_RATE = 0.02
DEFAULT_STUDENT_T_DF = 5.0


def compute_historical_stats(
    returns: pd.Series,
    freq: str,
    rf_rate: float = DEFAULT_RF_RATE,
) -> dict[str, float]:
    af = annualization_factor(freq)
    periodic_rf = rf_rate / af

    returns_clean = returns.dropna()
    obs = len(returns_clean)

    if obs < 2:
        return {f: 0.0 for f in ['mean', 'ann_mean', 'vol', 'ann_vol', 'skew', 'xkurt', 'var', 'etl', 'etr', 'sharpe', 'starr', 'rachev', 'max_dd', 'best_period', 'worst_period', 'obs_count']}

    mean_ret = float(returns_clean.mean())
    ann_mean = mean_ret * af
    vol = float(returns_clean.std(ddof=1))
    ann_vol = vol * np.sqrt(af)

    skew = float(stats.skew(returns_clean, bias=False))
    xkurt = float(stats.kurtosis(returns_clean, fisher=True, bias=False))

    left_q = 0.05
    right_q = 0.95
    var_val = float(-np.quantile(returns_clean, left_q))

    tail_returns = returns_clean[returns_clean <= np.quantile(returns_clean, left_q)]
    etl_val = float(-tail_returns.mean()) if len(tail_returns) > 0 else var_val

    right_tail_returns = returns_clean[returns_clean >= np.quantile(returns_clean, right_q)]
    etr_val = float(right_tail_returns.mean()) if len(right_tail_returns) > 0 else mean_ret

    sharpe = (mean_ret - periodic_rf) / vol if vol > 1e-10 else np.nan
    starr = (mean_ret - periodic_rf) / etl_val if etl_val > 1e-10 else np.nan
    rachev = etr_val / etl_val if etl_val > 1e-10 else np.nan

    cumsum = (1 + returns_clean).cumprod()
    running_max = cumsum.expanding().max()
    drawdown = cumsum / running_max - 1
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan

    best_period = float(returns_clean.max())
    worst_period = float(returns_clean.min())

    return {
        'mean': mean_ret,
        'ann_mean': ann_mean,
        'vol': vol,
        'ann_vol': ann_vol,
        'skew': skew,
        'xkurt': xkurt,
        'var': var_val,
        'etl': etl_val,
        'etr': etr_val,
        'sharpe': sharpe,
        'starr': starr,
        'rachev': rachev,
        'max_dd': max_dd,
        'best_period': best_period,
        'worst_period': worst_period,
        'obs_count': int(obs),
    }


def fit_student_t_distribution(returns: pd.Series) -> dict[str, float]:
    returns_clean = returns.dropna().values
    if len(returns_clean) < 10:
        return {'df': np.inf, 'loc': float(returns_clean.mean()), 'scale': float(returns_clean.std())}

    try:
        params = stats.t.fit(returns_clean)
        df, loc, scale = params
        df = max(2.0, float(df))
        return {'df': df, 'loc': float(loc), 'scale': float(scale)}
    except Exception:
        return {'df': 5.0, 'loc': float(returns_clean.mean()), 'scale': float(returns_clean.std())}


def simulate_fat_tailed_returns(
    returns_or_cov: pd.DataFrame | pd.Series,
    n_sims: int = DEFAULT_NUM_SIMULATIONS,
    random_seed: int | None = None,
) -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(returns_or_cov, pd.Series):
        t_params = fit_student_t_distribution(returns_or_cov)
        df, loc, scale = t_params['df'], t_params['loc'], t_params['scale']

        if np.isinf(df):
            sims = np.random.normal(loc, scale, n_sims)
        else:
            sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims)

        return pd.DataFrame({'Portfolio': sims})

    returns_df = returns_or_cov.copy()
    n_assets = returns_df.shape[1]

    t_params_list = [fit_student_t_distribution(returns_df[col]) for col in returns_df.columns]

    corr = returns_df.corr().fillna(0)
    try:
        L = np.linalg.cholesky(corr.values)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(np.eye(n_assets) * 0.99 + corr.values * 0.01)

    normal_sims = np.random.normal(0, 1, (n_sims, n_assets))
    correlated_normal = normal_sims @ L.T

    simulated_returns = np.zeros((n_sims, n_assets))
    for i, t_param in enumerate(t_params_list):
        df, loc, scale = t_param['df'], t_param['loc'], t_param['scale']
        if np.isinf(df):
            simulated_returns[:, i] = correlated_normal[:, i] * scale + loc
        else:
            u = stats.norm.cdf(correlated_normal[:, i])
            simulated_returns[:, i] = stats.t.ppf(u, df, loc=loc, scale=scale)

    return pd.DataFrame(simulated_returns, columns=returns_df.columns)


def compute_risk_metrics_from_dist(
    returns: np.ndarray | pd.Series,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float]:
    returns_clean = np.asarray(returns).flatten()
    returns_clean = returns_clean[~np.isnan(returns_clean)]

    left_q = 1 - confidence
    right_q = confidence

    var_val = float(-np.quantile(returns_clean, left_q))
    tail_mask = returns_clean <= np.quantile(returns_clean, left_q)
    etl_val = float(-returns_clean[tail_mask].mean()) if tail_mask.sum() > 0 else var_val

    right_tail_mask = returns_clean >= np.quantile(returns_clean, right_q)
    etr_val = float(returns_clean[right_tail_mask].mean()) if right_tail_mask.sum() > 0 else returns_clean.mean()

    return {'var': var_val, 'etl': etl_val, 'etr': etr_val}


def compute_marginal_risk_contributions(
    weights: np.ndarray,
    simulated_returns: pd.DataFrame,
    confidence: float = DEFAULT_CONFIDENCE,
    epsilon: float = 0.01,
) -> dict[str, np.ndarray]:
    n_assets = simulated_returns.shape[1]
    weights = np.asarray(weights).flatten()

    base_portfolio = simulated_returns.values @ weights
    base_vol = np.std(base_portfolio, ddof=1)
    base_metrics = compute_risk_metrics_from_dist(base_portfolio, confidence)
    base_etl = base_metrics['etl']
    base_etr = base_metrics['etr']

    mc_vol = np.zeros(n_assets)
    mc_etl = np.zeros(n_assets)
    mc_etr = np.zeros(n_assets)

    for i in range(n_assets):
        weights_perturbed = weights.copy()
        weights_perturbed[i] += epsilon
        weights_perturbed = weights_perturbed / weights_perturbed.sum()

        perturbed_portfolio = simulated_returns.values @ weights_perturbed
        new_vol = np.std(perturbed_portfolio, ddof=1)
        new_metrics = compute_risk_metrics_from_dist(perturbed_portfolio, confidence)
        new_etl = new_metrics['etl']
        new_etr = new_metrics['etr']

        mc_vol[i] = (new_vol - base_vol) / epsilon if base_vol > 1e-10 else 0.0
        mc_etl[i] = (new_etl - base_etl) / epsilon if base_etl > 1e-10 else 0.0
        mc_etr[i] = (new_etr - base_etr) / epsilon if base_etr > 1e-10 else 0.0

    return {'mc_vol': mc_vol, 'mc_etl': mc_etl, 'mc_etr': mc_etr}


def compute_percent_risk_contributions(
    weights: np.ndarray,
    marginal_contribs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    weights = np.asarray(weights).flatten()

    pc_vol = np.abs(weights * marginal_contribs['mc_vol'])
    pc_vol = pc_vol / (pc_vol.sum() + 1e-10)

    pc_etl = np.abs(weights * marginal_contribs['mc_etl'])
    pc_etl = pc_etl / (pc_etl.sum() + 1e-10)

    pc_etr = np.abs(weights * marginal_contribs['mc_etr'])
    pc_etr = pc_etr / (pc_etr.sum() + 1e-10)

    return {'pc_vol': pc_vol, 'pc_etl': pc_etl, 'pc_etr': pc_etr}


def compute_systematic_specific_risk(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    ols_model: Any | None = None,
) -> dict[str, float]:
    if ols_model is None:
        from .data import run_ols
        ols_model = run_ols(returns, factor_returns)['model']

    total_var = float(returns.var(ddof=1))

    betas = ols_model.params.drop(labels=['const'], errors='ignore')
    factor_cov = factor_returns.cov()
    systematic_var = float(np.dot(betas.values, np.dot(factor_cov.values, betas.values.T)))
    specific_var = float(ols_model.resid.var(ddof=1))

    total = systematic_var + specific_var
    systematic_pct = systematic_var / total if total > 0 else 0.0
    specific_pct = specific_var / total if total > 0 else 0.0

    return {
        'systematic_var': systematic_var,
        'specific_var': specific_var,
        'total_var': total_var,
        'systematic_pct': systematic_pct,
        'specific_pct': specific_pct,
    }


def compute_implied_returns(
    portfolio_mean: float,
    portfolio_risk: float,
    marginal_contributions: np.ndarray,
    rf_rate: float,
    af: int,
) -> np.ndarray:
    periodic_rf = rf_rate / af
    excess_return = portfolio_mean - periodic_rf

    if portfolio_risk < 1e-10:
        return np.full_like(marginal_contributions, portfolio_mean, dtype=float)

    performance_ratio = excess_return / portfolio_risk
    implied_excess = performance_ratio * marginal_contributions
    return implied_excess + periodic_rf


def compute_risk_budgeting_table(
    weights: np.ndarray,
    returns_matrix: pd.DataFrame,
    marginal_contribs: np.ndarray,
    risk_measure: str,
    rf_rate: float,
    af: int,
) -> pd.DataFrame:
    asset_names = returns_matrix.columns

    portfolio_rets = returns_matrix.values @ weights
    portfolio_mean = np.mean(portfolio_rets)
    if risk_measure == 'ETL':
        left_q = 0.05
        tail_mask = portfolio_rets <= np.quantile(portfolio_rets, left_q)
        portfolio_risk = np.abs(np.mean(portfolio_rets[tail_mask]))
    else:
        portfolio_risk = np.std(portfolio_rets, ddof=1)

    asset_means = returns_matrix.mean().values
    implied_rets = compute_implied_returns(portfolio_mean, portfolio_risk, marginal_contribs, rf_rate, af)
    differences = asset_means - implied_rets
    status = np.where(differences > 1e-4, '↑ Increase', np.where(differences < -1e-4, '↓ Decrease', '→ Neutral'))

    return pd.DataFrame(
        {
            'Asset': asset_names,
            'Weight (%)': weights * 100,
            'Mean Return (%)': asset_means * 100,
            'MC to Risk': marginal_contribs,
            'Implied Return (%)': implied_rets * 100,
            'Difference (%)': differences * 100,
            'Status': status,
        }
    )


def get_factor_bucket_mapping(factor_names: list[str]) -> dict[str, list[str]]:
    buckets = {
        'Equity Risk': [],
        'Commodity Risk': [],
        'FX Risk': [],
        'Interest Rate Risk': [],
        'Sovereign Risk': [],
        'Corporate Risk': [],
        'Volatility Risk': [],
        'Fixed Income Risk': [],
    }

    keywords = {
        'Equity Risk': ['msci', 'russell', 's&p', 'equity', 'value', 'growth', 'small', 'large', 'world', 'eafe', 'em'],
        'Commodity Risk': ['gsci', 'crude', 'oil', 'commodity', 'gold', 'copper', 'agriculture', 'metal'],
        'FX Risk': ['dollar', 'euro', 'yen', 'sterling', 'pound', 'franc', 'krona', 'renminbi', 'fx', 'exchange'],
        'Interest Rate Risk': ['rate', 'libor', 'sofr', 'yield', 'spread', 'maturity', 'treasury', 'duration'],
        'Sovereign Risk': ['sovereign', 'government bond', 'treasury', 'master'],
        'Corporate Risk': ['corporate', 'high yield', 'crossover', 'credit', 'bbb', 'aaa'],
        'Volatility Risk': ['vix', 'cboe', 'volatility'],
        'Fixed Income Risk': ['bond', 'mortgage', 'abs', 'mbs', 'cmbs', 'fixed income'],
    }

    for factor in factor_names:
        factor_lower = factor.lower()
        matched = False
        for bucket, kws in keywords.items():
            if any(kw in factor_lower for kw in kws):
                buckets[bucket].append(factor)
                matched = True
                break
        if not matched:
            buckets.setdefault('Other Risk', []).append(factor)

    return {k: v for k, v in buckets.items() if v}


def compute_factor_contribution(
    ols_model: Any,
    factor_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    simulated_portfolio_returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    betas = ols_model.params.drop(labels=['const'], errors='ignore')
    factor_names = betas.index.tolist()

    factor_cov = factor_returns.cov()
    residuals = ols_model.resid
    total_var = portfolio_returns.var(ddof=1)

    systematic_var = float(np.dot(betas.values, np.dot(factor_cov.values, betas.values.T)))
    specific_var = float(residuals.var(ddof=1))

    systematic_stdev = np.sqrt(max(systematic_var, 0))
    specific_stdev = np.sqrt(max(specific_var, 0))
    total_stdev = np.sqrt(total_var)

    systematic_pct = systematic_stdev / total_stdev if total_stdev > 1e-10 else 0.0
    specific_pct = specific_stdev / total_stdev if total_stdev > 1e-10 else 0.0

    factor_mc_stdev = np.zeros(len(factor_names))
    for i, _factor in enumerate(factor_names):
        factor_mc_stdev[i] = betas.iloc[i] * factor_cov.iloc[i, i] / (systematic_stdev + 1e-10) if systematic_stdev > 1e-10 else 0.0

    factor_contrib_etl = np.abs(factor_mc_stdev) * (np.std(simulated_portfolio_returns) / (total_stdev + 1e-10))
    bucket_mapping = get_factor_bucket_mapping(factor_names)

    return {
        'systematic_stdev_pct': systematic_pct,
        'specific_stdev_pct': specific_pct,
        'betas': betas,
        'factor_names': factor_names,
        'factor_mc_stdev': factor_mc_stdev,
        'factor_contrib_etl': factor_contrib_etl,
        'bucket_mapping': bucket_mapping,
    }


def compute_factor_bucket_exposures(
    betas: pd.Series,
    bucket_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []

    for bucket, factors in bucket_mapping.items():
        matching_betas = [betas.get(f, 0.0) for f in factors if f in betas.index]
        if not matching_betas:
            continue

        rows.append(
            {
                'Bucket': bucket,
                'Exposure': float(sum(matching_betas)),
                'Factor Count': len(matching_betas),
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()
