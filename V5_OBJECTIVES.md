# V5 Objective Notes

## Question 1.1

Every signal ablation reports two outcomes:

- mean price and markup, indicating whether supracompetitive pricing changed;
- state-conditional CMI, its permutation z-score, flag share, and action
  entropy, indicating whether pairwise dependence changed.

The public conditioning state is the previous round's mean price, binned at
`1.10, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75`. Actions are own-price
down, hold, or up moves with a `0.01` hold band.

The build passes two positive controls. Calibrated agents have higher markup
and CMI than the hard ban. Scripted colluders have higher markup and CMI than
competitive episodes. The strongest ablations of precision, delay, and
visibility all reduce both outcomes. A one-time `history_reset` at round 50
with a `1.473` warm-up price increases markup by `4.9%` and reduces CMI by
`4.1%` in this model.

## Question 1.5

The restriction ranking requires:

- a passing positive control;
- a positive lower 90% confidence bound on markup reduction;
- a positive lower 90% confidence bound on CMI reduction;
- no frozen-price artifact;
- exclusion of the hard ban from the signal-restriction ranking.

Eligible rows are ordered by welfare efficiency:

```text
markup reduction / absolute welfare movement
```

The hard ban remains visible as a comparator but cannot win the signal-
restriction ranking.

## Current Result

Precision degradation at `0.40` is the selected restriction:

- mean price falls from `1.4146` to `1.1077`;
- mean markup falls by `74.0%`;
- CMI falls by `85.4%`;
- action entropy remains `1.55` bits, so the result is not a freeze artifact;
- the welfare effect is a `0.53%` improvement;
- efficiency ratio is `0.248` with a 90% interval of `[0.248, 0.673]`.

The hard ban reduces markup by `39.6%`, reduces CMI by `93.7%`, and costs
`3.34%` of welfare in this simulation. It is less efficient than precision
degradation.

History reset does not beat continuous noise injection. It leaves mean markup
at `0.435` and CMI at `0.392` bits. The direct ranking is written to both
`results/objective_1_5_restriction_ranking.csv` and
`results/restriction_ranking.csv`.
