# Model Roadmap

This document records the next planned model-sophistication upgrades for RiskPlus Streamlit. The goal is to make future math improvements easier without changing any current outputs today.

## Current State

The current pipeline is intentionally stable. It produces the existing historical, simulated, factor, risk-budgeting, and data-quality outputs from the uploaded data and current open-source formulas.

## Planned Enhancements

### 1. Gaussian vs. Student-t simulation wiring

The UI already exposes distribution choice, but the roadmap item is to ensure the selected distribution is fully wired through the simulation engine with explicit parity checks and documented assumptions.

### 2. EWMA covariance

Add a dedicated covariance path that uses exponentially weighted returns when selected, while preserving the current classical covariance behavior as the default.

### 3. Historical covariance vs. shrinkage covariance

Add support for choosing between plain historical covariance and a shrinkage estimator. This will help stabilize factor and portfolio risk estimates when samples are short or noisy.

### 4. Fund-level factor models

Extend factor analysis from portfolio-level regression toward fund-level diagnostics so each fund can be explained against the selected factor set.

### 5. Rolling beta

Add rolling beta estimation so users can see how factor sensitivity changes across time rather than only over the full overlap window.

### 6. Stepwise AIC/BIC factor selection

Introduce optional factor selection diagnostics using AIC/BIC stepwise routines. This should remain optional so current factor selections and outputs do not change by default.

### 7. Full factor covariance contribution

Add a more explicit decomposition of factor covariance contributions so users can see how each factor pair affects total risk.

### 8. Scenario and stress testing

Add reusable scenario objects for shocks, regime shifts, and factor drawdowns. These should be read-only analysis tools until a future release wires them into the UI.

### 9. Optimization and what-if portfolios

Add a future optimization layer for target-risk or target-return what-if portfolios. This should remain separate from the current reporting path.

### 10. Excel / PDF export

Add export interfaces for workbook and PDF reporting so the current views can be distributed without changing calculations.

## Guardrails

- Do not change current formulas unless a future prompt explicitly asks for it.
- Do not alter existing report values or table shapes by default.
- Prefer additive interfaces and feature flags over in-place rewrites.
- Keep the current upload modes and report flow stable while future math work is layered on top.

## Suggested Sequence

1. Wire simulation distribution choice end to end.
2. Add covariance strategy interfaces.
3. Add rolling factor diagnostics.
4. Add scenario/stress abstractions.
5. Add export interfaces last, after the report content is stable.
