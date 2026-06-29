from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from riskplus_core.baseline import run_baseline_comparison


def main() -> None:
    comparisons = run_baseline_comparison(num_sims=10_000, random_seed=42)
    print('Wrote baseline comparison reports:')
    for name, frame in comparisons.items():
        print(f'- {name}: {len(frame)} rows')


if __name__ == '__main__':
    main()