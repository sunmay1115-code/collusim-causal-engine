Use this as `README.md`:

```markdown
# Collusim V9

**Track 1: measuring and restricting collusion between LLM market agents**

Collusim is a deterministic research stack for testing market restrictions under
conditions where LLM agents hide intent, freeze prices, or inherit collusive
behavior from shared history.

V9 preserves the calibrated CMI detector, Collusim RCT, objective gates, and
two-tab dashboard from the core V7 build. It adds a standalone live-noise stress
test that measures whether realistic LLM stochasticity weakens causal policy
comparisons.

[Read the report](report/report.pdf) |
[Architecture](V5_ARCHITECTURE.md) |
[Objectives](V5_OBJECTIVES.md) |
[Restriction ranking](results/objective_1_5_restriction_ranking.csv)

## What V9 Adds

- A `LiveNoiseAgent` that adds empirical residuals from local live-model runs
  to the calibrated OLS policy.
- A 20-seed paired comparison of three conditions: OLS baseline,
  noise injection at strength 0.40, and a hard ban.
- Bootstrap confidence intervals for markup effects and paired contrasts.
- A saved live-noise Pareto figure and machine-readable summary.
- Explicit separation between behavioral observations and causal simulation.

The V9 experiment uses saved local Qwen2.5-1.5B data. It does not claim results
from hosted GPT-4.1 or Claude runs.

## Headline Results

### Core policy ranking

| Restriction | Markup reduction | CMI reduction | Result |
|---|---:|---:|---|
| **Precision loss, 0.40** | **74.0%** | **85.4%** | **Rank 1** |
| History reset at round 50 | -4.9% | 4.1% | Fails material-effect gate |
| Hard ban | 39.6% | 93.7% | Ineligible signal restriction |

Precision loss is the highest-efficiency eligible restriction. It reduces
simulated markup by 74.0%, reduces CMI by 85.4%, improves simulated welfare by
0.53%, and avoids a frozen-price artifact. The history reset does not beat
continuous noise injection in the calibrated simulator.

### V9 live-noise stress test

The saved V9 dataset contains 1,353 valid local Qwen2.5-1.5B observations. The
estimated residual standard deviation is **0.2983**.

| Condition | Mean markup | Total welfare | Mean CMI bits |
|---|---:|---:|---:|
| OLS baseline | 0.690 | 226.42 | 0.423 |
| Noise injection 0.40 | **0.210** | **232.07** | 0.063 |
| Hard ban | 0.251 | 224.63 | 0.026 |

Across 20 paired seeds:

- Noise 0.40 reduces markup by **48.1%**
  (90% CI: 44.2% to 52.1%).
- The hard ban reduces markup by **44.0%**
  (90% CI: 39.8% to 48.5%).
- The paired markup contrast is **-4.08 percentage points**
  (90% CI: -4.72 to -3.42), where noise injection is below the hard ban.

![V9 live-noise policy frontier](results/figures/live_noise_frontier.png)

The injected noise makes the policy frontier more diffuse and reduces the
separation between interventions. This motivates controlled simulation for
causal comparisons while retaining live-model data for behavioral calibration
and residual estimation.

## Evidence Stack

1. **Empirical motivation.** The project uses Assanali's 75-run E3 auction
   evidence, the E4 intent-gap test, and Abilmansur's shared-history replacement
   experiment.
2. **Detector.** The `infocollusion` CMI wrapper conditions on public market
   state, uses Miller-Madow estimation with a within-state permutation test, and
   corrects across agent pairs.
3. **Simulation.** Collusim runs paired-seed restrictions for precision loss,
   price delay, visibility loss, shared-history reset, and a hard ban.
4. **Objective gates.** Restrictions must pass positive controls, confidence
   bounds, CMI-reduction, and freeze-status checks before ranking.
5. **Starter validation.** One-seed official-starter runs validate integration
   through a deterministic local OpenAI-compatible mock. They are not live LLM
   experiments.
6. **Robustness.** The project includes E3 detector consistency, CMI bin
   sensitivity, welfare-weight sensitivity, and lagged-CMI comparisons.
7. **V9 live noise.** Empirical local-model residuals are injected into the OLS
   proxy and tested across 20 paired seeds.

## Repository Map

```text
app.py                          Two-tab Streamlit dashboard
run_v7_pipeline.py              Core deterministic pipeline
src/collusim_noise_test.py      V9 live-noise ablation
src/collusim/                   Simulator, agents, restrictions, objectives
src/detection/                  CMI wrapper
src/v6_robustness.py            Robustness checks
tools/                          Official-starter validation
tests/                          Unit and integration tests
results/                        Saved metrics, CSVs, JSON, and figures
report/                         Research report and figure builders
data/experimental/              Local live-model residual dataset
```

## Reproduce V9

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Rebuild the core deterministic pipeline:

```bash
python run_v7_pipeline.py
```

Run the standalone V9 live-noise ablation:

```bash
python -m src.collusim_noise_test
```

Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard reads saved artifacts and does not contact a model provider.

## Tests

```bash
python -m pytest tests -q
```

V9-specific checks verify the empirical noise estimate, the three paired
conditions, the 20-seed structure, the summary JSON, and the saved figure.

## Key Outputs

- `results/live_noise_ablation.csv`: seed-level V9 outcomes.
- `results/live_noise_ablation_summary.json`: confidence intervals and
  condition means.
- `results/figures/live_noise_frontier.png`: V9 frontier figure.
- `results/objective_1_5_restriction_ranking.csv`: core restriction ranking.
- `results/collusim_rct_raw.csv`: seed-level core RCT outcomes.
- `results/lagged_cmi_comparison.csv`: contemporaneous and lagged CMI results.
- `results/v6_robustness_summary.json`: bin, E3, and welfare robustness results.
- `results/e3_consistency.csv`: real E3 wrapper consistency results.
- `report/report.pdf`: final research report.

## Scientific Boundaries

- Collusim is a calibrated two-agent simulator, not a population of live
  GPT-4.1 or Claude sellers.
- The circuit-breaker detector is validated on a synthetic scripted benchmark,
  not live LLM markets.
- The CMI wrapper is anchored against 75 real E3 auction logs, but remains a
  coordination screen rather than a legal cartel verdict.
- Starter-code checks validate execution paths through a deterministic local
  mock and do not estimate frontier-model behavior.
- The hard ban is a no-LLM comparator, not a recommended production policy.
- V9 adds empirical noise to the proxy; it is not a full live multi-agent RCT.

## Team

- **Man Ki Kuen (Sunny):** Collusim integration, objective ranking, dashboard,
  robustness checks, and V9 live-noise ablation.
- **Assanali (Assan) Alibatyrkhan:** calibrated CMI detector, E3 stasis
  evidence, and E4 intent-gap evidence.
- **Abilmansur Alimbayev:** replacement study and shared-history evidence.

Built for the AI Collusion Research Sprint by Apart Research and Poseidon
Research, October 2026.
```
