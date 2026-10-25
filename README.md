# Collusim Causal Engine

**Track 1: measuring and restricting collusion between LLM market agents**

V5 is a deterministic research stack for connecting empirical failures of LLM
collusion detection to a controlled restriction experiment. It combines a
calibrated conditional mutual information (CMI) detector, Collusim's
paired-seed RCT, strict objective gates, and a two-tab dashboard.

[Read the report](report/report.pdf) | [Architecture](V5_ARCHITECTURE.md) |
[Objective definition](V5_OBJECTIVES.md) |
[Restriction ranking](results/objective_1_5_restriction_ranking.csv)

## Headline Result

| Restriction | Markup reduction | CMI reduction | Welfare change | Rank |
|---|---:|---:|---:|---:|
| **Precision loss, strength 0.40** | **74.0%** | **85.4%** | **+0.53%** | **1** |
| History reset, round 50 | -4.9% | 4.1% | +0.01% | 13 |
| Hard ban | 39.6% | 93.7% | +3.34% | Ineligible as a signal restriction |

Precision loss wins the strict objective ranking because both the markup and
CMI reductions clear their confidence-bound gates without a frozen-price
artifact. The specified history reset does not beat continuous noise injection
in V5.

## Evidence Stack

1. **The Problem.** Three empirical findings establish the detection and
   enforcement failures:
   - In 75 GPT-4.1/Claude auctions, colluding sellers held prices in 87% of
     rounds.
   - In 12 Qwen2.5-7B games, an explicit collusion instruction produced no
     reliable pair-specific behavioral difference.
   - Replacing a cartel member did not break the cartel: the newcomer inherited
     the shared price history (profit gain 0.80 versus 0.32 for a fresh pair).
2. **The Detector.** The `infocollusion` CMI wrapper conditions on the previous
   public mean-price bin, applies Miller-Madow estimation with a within-state
   permutation test, and corrects across agent pairs.
3. **The Solution.** Collusim runs paired-seed counterfactuals for precision
   loss, delay, visibility loss, history reset, and a hard ban. Objective gates
   reject weak effects, unstable controls, and detection artifacts caused by
   price stasis.
4. **Starter Validation.** One-seed ablations using the official Q1.1 and Q1.5
   starter CLIs validate integration. They used a deterministic local
   OpenAI-compatible mock because no hosted API key was configured; these runs
   are not presented as live LLM evidence.

## Repository Map

```text
app.py                     Two-tab Streamlit dashboard
run_v5_pipeline.py         Deterministic detector and Collusim pipeline
src/collusim/              Simulator, restrictions, and objective gates
src/detection/             Teammate's unchanged CMI wrapper
src/external/              Adapted detector benchmark components
tools/                     Official-starter validation harness
tests/                     Detector, simulator, objective, and starter tests
results/                   Final CSV, JSON, model, and figure artifacts
report/report.pdf          Final research report
data/                      Local data mount points
```

## Quick Start

```bash
python -m pip install -r requirements.txt
python run_v5_pipeline.py
streamlit run app.py
```

The dashboard reads saved artifacts and does not contact a model provider.

Validation:

```bash
python -m pytest tests -q
python tools/run_starter_validation.py
```

## Key Outputs

- `results/collusim_rct_raw.csv`: seed-level RCT outcomes.
- `results/collusim_frontier.csv`: aggregate restriction frontier.
- `results/objective_1_1_variable_ablations.csv`: objective 1.1 evidence.
- `results/objective_1_5_restriction_ranking.csv`: objective 1.5 ranking.
- `results/starter_q11_baseline.csv` and `starter_q11_ablation.csv`: history
  ablation.
- `results/starter_q15_baseline.csv` and `results/starter_q15_ablation.csv`:
  message-cap ablation.
- `results/v5_objective_summary.json`: machine-readable result values.

## Scientific Boundary

Collusim is a calibrated two-agent simulator, not a population of GPT-4.1 or
Claude sellers. Its causal result is conditional on the simulated pricing
policy and market dynamics. The CMI value is a coordination screen, not a
stand-alone legal or operational cartel verdict. The project keeps empirical
evidence and simulator evidence explicitly separate.

## Team

- **Man Ki Kuen (Sunny):** V5 integration, Collusim RCT, objective gates,
  dashboard, and starter validation.
- **Assanali (Assan) Alibatyrkhan:** calibrated CMI detector, E3 stasis
  evidence, and E4 intent-gap evidence.
- **Abilmansur Alimbayev:** replacement experiment and shared-history evidence.

Built for the AI Collusion Research Sprint by Apart Research and Poseidon
Research, October 2026.
