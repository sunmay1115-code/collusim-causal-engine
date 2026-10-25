# LLM Pricing Behavior Dataset

- Version: `1.0`
- Run: `synthetic_20261024T124450Z`
- Source: `synthetic_validation`
- Models: llama3:8b, qwen2.5:7b
- Requested states per model: 500
- Raw rows: 1000
- Valid rows: 1000
- Failed rows: 0
- Temperature: 0.2

## Unit of observation

One row is one market state paired with one LLM prompt-response.
The same state ids are used across models. The analysis-ready file
contains valid responses only; the raw file retains failures and
invalid outputs.

## Labels

`strategy` is derived from `llm_price` and `competitor_price`
with epsilon `0.05` as undercut, match, or overcut.

## Classification

### Valid rows by model

```json
{
  "llama3:8b": 500,
  "qwen2.5:7b": 500
}
```

### Strategy counts

```json
{
  "llama3:8b": {
    "match": 265,
    "overcut": 130,
    "undercut": 105
  },
  "qwen2.5:7b": {
    "match": 208,
    "overcut": 166,
    "undercut": 126
  }
}
```

## Intended use

This dataset is the primary artifact. Regressions, hypothesis tests,
predictive models, SHAP, and calibration are validation layers over
this file, not substitutes for it.