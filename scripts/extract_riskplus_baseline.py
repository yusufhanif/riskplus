from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from riskplus_core.baseline import extract_riskplus_baseline


def main() -> None:
    artifacts = extract_riskplus_baseline()
    print('Extracted RiskPlus baseline fixtures:')
    print(f'- settings: {len(artifacts.settings)} rows')
    print(f'- historical risk: {len(artifacts.historical_risk)} rows')
    print(f'- simulated risk: {len(artifacts.simulated_risk)} rows')
    print(f'- summary data: {len(artifacts.summary_data)} rows')
    print(f'- rb_etl: {len(artifacts.rb_etl)} rows')
    print(f'- rb_stdev: {len(artifacts.rb_stdev)} rows')
    print(f'- factor analysis: {len(artifacts.factor_analysis)} rows')
    print(f'- factor correlations: {len(artifacts.factor_correlations)} rows')
    print(f'- asset correlations: {len(artifacts.asset_correlations)} rows')


if __name__ == '__main__':
    main()