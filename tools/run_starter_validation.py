"""Run one-seed offline proof-of-concept checks through official starter CLIs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pandas as pd

from starter_mock_server import make_server


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STARTER = ROOT / "tmp" / "collusion_hackathon"


def _env(port: int, starter: Path, project: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": os.pathsep.join(
                [
                    str(starter / "util"),
                    str(project),
                ]
            ),
            "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "LLM_API_KEY": "offline-starter-mock",
            "LLM_PROVIDER": "openai-compatible",
            "MODEL_NAME": "starter-mock",
        }
    )
    return env


def _run(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(command)}\n\n{result.stdout}\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed; see {log_path}")


def _newest_run(parent: Path, started: float) -> Path:
    candidates = [
        path.parent
        for path in parent.rglob("metadata.json")
        if path.stat().st_mtime >= started - 1
    ]
    if not candidates:
        raise FileNotFoundError(f"No new official run found under {parent}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _q11_trajectory(run_dir: Path, visible_rounds: int) -> pd.DataFrame:
    history = pd.read_parquet(run_dir / "environment_history.parquet")
    history = history.loc[history["agent_type"] == "LLM_agent"]
    import json

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    benchmark_prices = [
        float(agent["nash_price"])
        for agent in metadata["summary"]["agents"].values()
    ]
    summary = (
        history.groupby("round", as_index=False)
        .agg(mean_price=("price", "mean"), agents=("agent", "nunique"))
    )
    pivot = history.pivot_table(
        index="round",
        columns="agent",
        values="price",
    ).sort_index()
    pivot.columns = [
        f"agent_{index + 1}_price" for index in range(len(pivot.columns))
    ]
    summary = summary.merge(pivot.reset_index(), on="round", how="left")
    summary.insert(1, "visible_rounds", visible_rounds)
    summary.insert(
        2,
        "competitive_benchmark",
        sum(benchmark_prices) / len(benchmark_prices),
    )
    summary["source"] = "official_duopoly_starter_local_mock"
    return summary


def _q15_metrics(run_root: Path) -> dict:
    metrics_paths = sorted(run_root.rglob("collusion_metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics found under {run_root}")
    import json

    return json.loads(metrics_paths[-1].read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-root", type=Path, default=STARTER)
    parser.add_argument("--rounds-q11", type=int, default=30)
    parser.add_argument("--rounds-q15", type=int, default=15)
    args = parser.parse_args()
    starter = args.starter_root.resolve()
    algorithmic = starter / "algorithmic-pricing-llms"
    auction = starter / "llm-agent-collusion"
    if not algorithmic.exists() or not auction.exists():
        raise FileNotFoundError("Official starter checkout is incomplete.")

    server = make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    q11_env = _env(server.server_port, starter, algorithmic)
    q15_env = _env(server.server_port, starter, auction)
    raw = RESULTS / "starter_runs"
    raw.mkdir(parents=True, exist_ok=True)

    try:
        q11_runs = algorithmic / "experiments_synthetic" / "experiments_runs" / "2_agents"
        started = time.time()
        _run(
            [
                sys.executable,
                "experiments_synthetic/duopoly.py",
                "--rounds",
                str(args.rounds_q11),
                "--memory",
                "100",
                "--seeds",
                "0",
                "--provider",
                "openai-compatible",
                "--model",
                "starter-mock",
                "--plot-every",
                "0",
                "-y",
            ],
            algorithmic,
            q11_env,
            raw / "q11_baseline.log",
        )
        baseline_dir = _newest_run(q11_runs, started)
        baseline_q11 = _q11_trajectory(baseline_dir, 100)
        baseline_q11.to_csv(
            RESULTS / "starter_q11_baseline.csv",
            index=False,
        )
        print(
            "Q1.1 baseline average price: "
            f"{baseline_q11['mean_price'].mean():.4f}; "
            f"competitive benchmark: "
            f"{baseline_q11['competitive_benchmark'].iloc[0]:.4f}"
        )

        started = time.time()
        _run(
            [
                sys.executable,
                "experiments_synthetic/duopoly.py",
                "--rounds",
                str(args.rounds_q11),
                "--memory",
                "0",
                "--seeds",
                "0",
                "--provider",
                "openai-compatible",
                "--model",
                "starter-mock",
                "--plot-every",
                "0",
                "-y",
            ],
            algorithmic,
            q11_env,
            raw / "q11_ablation.log",
        )
        ablation_dir = _newest_run(q11_runs, started)
        ablation_q11 = _q11_trajectory(ablation_dir, 0)
        ablation_q11.to_csv(
            RESULTS / "starter_q11_ablation.csv",
            index=False,
        )
        print(
            "Q1.1 ablation average price: "
            f"{ablation_q11['mean_price'].mean():.4f}; "
            f"competitive benchmark: "
            f"{ablation_q11['competitive_benchmark'].iloc[0]:.4f}"
        )

        for label, cap in (("baseline", 500), ("ablation", 10)):
            run_root = raw / "q15" / label
            started = time.time()
            _run(
                [
                    sys.executable,
                    "src/continuous_double_auction/simulation.py",
                    "--rounds",
                    str(args.rounds_q15),
                    "--sellers",
                    "3",
                    "--buyers",
                    "3",
                    "--seeds",
                    "0",
                    "--seller_models",
                    "starter-mock",
                    "--buyer_models",
                    "starter-mock",
                    "--seller_comms_enabled",
                    "--max_message_length",
                    str(cap),
                    "--tag",
                    f"starter_q15_{label}",
                    "--log_dir",
                    str(run_root),
                    "-y",
                ],
                auction,
                q15_env,
                raw / f"q15_{label}.log",
            )
            metrics = _q15_metrics(run_root)
            result = {
                "average_price": metrics.get("avg_trade_price_overall"),
                "seller_profit": metrics.get("combined_seller_profits"),
                "trade_volume": metrics.get("num_trades"),
                "buyer_welfare": metrics.get("combined_buyer_surplus"),
            }
            print(f"Q1.5 {label} metrics: {result}")
            pd.DataFrame([result]).to_csv(
                RESULTS / f"starter_q15_{label}.csv",
                index=False,
            )
        combined_path = RESULTS / "starter_q15_results.csv"
        combined_path.unlink(missing_ok=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    print("Starter validation artifacts written under results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
