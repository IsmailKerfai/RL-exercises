from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl_exercises.week_3.td_lambda import run_td_lambda_experiment


def main() -> None:
    rows = run_td_lambda_experiment()
    outdir = Path("results") / "td_lambda" / "random_walk" / "seed_0"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "lambda_results.csv", index=False)

    for row in rows:
        print(f"lambda={row['lambda']:.1f}, rmse={row['rmse']:.4f}")


if __name__ == "__main__":
    main()
