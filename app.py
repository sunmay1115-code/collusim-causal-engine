"""Two-tab dashboard for V5 objectives 1.1 and 1.5."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import RESULTS_DIR


st.set_page_config(
    page_title="Collusim v5",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="ascii"))


def starter_q11_figure(
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.2, 3.2))
    axis.plot(
        baseline["round"],
        baseline["mean_price"],
        color="#0F766E",
        linewidth=1.9,
        marker="o",
        markersize=3,
        label="100 visible rounds",
    )
    axis.plot(
        ablation["round"],
        ablation["mean_price"],
        color="#C2410C",
        linewidth=1.9,
        marker="o",
        markersize=3,
        label="0 visible rounds",
    )
    axis.set_xlabel("Round")
    axis.set_ylabel("Mean price")
    axis.set_title("Starter Code Replication (Visible Rounds)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def frontier_figure(frontier: pd.DataFrame) -> tuple[plt.Figure, pd.Series]:
    colors = {
        "precision": "#0F766E",
        "delay": "#D97706",
        "visibility": "#7C3AED",
        "history_reset": "#2563EB",
    }
    labels = {
        "precision": "Noise injection",
        "delay": "Delay",
        "visibility": "Visibility loss",
        "history_reset": "History reset",
    }
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for lever, part in frontier.groupby("lever"):
        part = part.sort_values("strength")
        ax.plot(
            part["welfare_cost"],
            part["collusion_reduction"],
            marker="o",
            linewidth=2,
            color=colors[lever],
            label=labels[lever],
        )
    best = frontier.sort_values("efficiency_ratio", ascending=False).iloc[0]
    ax.scatter(
        [best["welfare_cost"]],
        [best["collusion_reduction"]],
        marker="*",
        s=240,
        color="#F59E0B",
        edgecolor="#78350F",
        zorder=5,
        label="selected restriction",
    )
    ax.axhline(0, color="#64748B", linewidth=0.8)
    ax.axvline(0, color="#64748B", linewidth=0.8)
    ax.set_xlabel("Welfare impact (positive = cost, negative = gain)")
    ax.set_ylabel("Collusion reduction")
    ax.set_title("Collusim randomized restriction frontier")
    ax.grid(alpha=0.2)
    ax.legend(
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    return fig, best


def two_metric_figure(
    frontier: pd.DataFrame,
    baseline_markup: float,
    baseline_cmi: float,
) -> plt.Figure:
    ordered = frontier.sort_values(["lever", "strength"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = {
        "precision": "#0F766E",
        "delay": "#D97706",
        "visibility": "#7C3AED",
        "history_reset": "#2563EB",
        "hard_ban": "#64748B",
    }
    labels = {
        "precision": "Noise injection",
        "delay": "Delay",
        "visibility": "Visibility",
        "history_reset": "History reset",
        "hard_ban": "Ban",
    }
    figure, (price_axis, cmi_axis) = plt.subplots(
        2,
        1,
        figsize=(11, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    for lever, part in ordered.groupby("lever", sort=False):
        indices = part.index.to_numpy()
        color = colors.get(lever, "#334155")
        price_axis.plot(
            indices,
            part["mean_markup"],
            color=color,
            marker="o",
            linewidth=2,
            label=labels.get(lever, lever),
        )
        cmi_axis.plot(
            indices,
            part["cmi_bits"],
            color=color,
            marker="o",
            linewidth=2,
            label=labels.get(lever, lever),
        )
        for row_index, row in part.iterrows():
            status = str(row["restriction_status"])
            marker = (
                "X"
                if status.startswith("Freeze")
                else "s"
                if status.startswith("Broken")
                else "o"
            )
            cmi_axis.scatter(
                [row_index],
                [row["cmi_bits"]],
                marker=marker,
                s=110 if marker != "o" else 54,
                color=color,
                edgecolor="#FFFFFF",
                linewidth=1,
                zorder=4,
            )
    price_axis.axhline(
        baseline_markup,
        color="#94A3B8",
        linestyle="--",
        linewidth=1,
        label="Baseline price level",
    )
    cmi_axis.axhline(
        baseline_cmi,
        color="#94A3B8",
        linestyle="--",
        linewidth=1,
        label="Baseline CMI",
    )
    price_axis.set_ylabel("Mean markup")
    price_axis.set_title("Price Level: high means supracompetitive pricing")
    price_axis.grid(alpha=0.2)
    price_axis.legend(frameon=False, ncol=6, loc="upper right")
    cmi_axis.set_ylabel("CMI (bits)")
    cmi_axis.set_title(
        "Coordination signal: X = freeze / failed, square = broken / successful"
    )
    cmi_axis.grid(alpha=0.2)
    cmi_axis.set_xticks(
        x,
        [
            f"{labels.get(row.lever, row.lever)}\n{row.strength:g}"
            for row in ordered.itertuples()
        ],
        rotation=45,
        ha="right",
    )
    figure.tight_layout()
    return figure


required = [
    "collusim_frontier.csv",
    "collusim_rct_raw.csv",
    "objective_1_1_positive_control.csv",
    "objective_1_5_restriction_ranking.csv",
    "v5_objective_summary.json",
]
if not all((RESULTS_DIR / name).exists() for name in required):
    st.error("Collusim v5 results are missing. Run `python run_v5_pipeline.py`.")
    st.stop()


frontier = load_csv("collusim_frontier.csv")
signal_frontier = frontier.loc[
    frontier["lever"].isin(
        ["precision", "delay", "visibility", "history_reset"]
    )
].copy()
objective_controls = load_csv("objective_1_1_positive_control.csv")
objective_ranking = load_csv("objective_1_5_restriction_ranking.csv")
objective_summary = load_json("v5_objective_summary.json")
raw = load_csv("collusim_rct_raw.csv")
q11_baseline_path = RESULTS_DIR / "starter_q11_baseline.csv"
q11_ablation_path = RESULTS_DIR / "starter_q11_ablation.csv"
q15_baseline_path = RESULTS_DIR / "starter_q15_baseline.csv"
q15_ablation_path = RESULTS_DIR / "starter_q15_ablation.csv"
q11_baseline = (
    load_csv(q11_baseline_path.name) if q11_baseline_path.exists() else None
)
q11_ablation = (
    load_csv(q11_ablation_path.name) if q11_ablation_path.exists() else None
)
q15_baseline = (
    load_csv(q15_baseline_path.name) if q15_baseline_path.exists() else None
)
q15_ablation = (
    load_csv(q15_ablation_path.name) if q15_ablation_path.exists() else None
)

baseline_markup = float(
    raw.loc[raw["lever"] == "baseline", "mean_markup"].mean()
)
baseline_cmi = (
    float(raw.loc[raw["lever"] == "baseline", "cmi_bits"].mean())
    if "cmi_bits" in raw.columns
    else 0.0
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17211F;
        --muted: #5F6B68;
        --rule: #DCE4E1;
        --paper: #FFFFFF;
        --canvas: #F3F6F5;
        --teal: #0F766E;
        --red: #C2410C;
        --amber: #D97706;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    html, body, [class*="css"] {
        font-family: Aptos, "Segoe UI", sans-serif;
        letter-spacing: 0;
    }
    .block-container {
        max-width: 1420px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    .hero {
        border-bottom: 1px solid var(--rule);
        padding-bottom: 1rem;
        margin-bottom: 0.9rem;
    }
    .kicker {
        color: var(--teal);
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: var(--ink);
        font-size: clamp(2.4rem, 4vw, 3.5rem);
        line-height: 1;
        margin: 0.35rem 0 0.55rem 0;
    }
    .hero p { color: var(--muted); font-size: 1.05rem; line-height: 1.5; }
    .hero .subtitle {
        color: var(--ink);
        font-size: 1.22rem;
        font-weight: 650;
        max-width: 980px;
        line-height: 1.45;
        margin: 0.15rem 0 0.85rem 0;
    }
    .badge {
        display: inline-block;
        padding: 0.32rem 0.58rem;
        border-radius: 999px;
        background: #FEF3C7;
        color: #92400E;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.7rem;
    }
    .big-number {
        background: var(--paper);
        border-left: 6px solid var(--teal);
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin: 0.35rem 0 1rem 0;
    }
    .big-number .number {
        color: var(--ink);
        font-size: 2.7rem;
        font-weight: 780;
        line-height: 1.08;
    }
    .big-number .copy {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.45;
    }
    .explain {
        background: var(--paper);
        border: 1px solid var(--rule);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: var(--ink);
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0.4rem 0 0.8rem 0;
    }
    .finding-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 1.2rem 0 1rem 0;
    }
    .finding-card {
        background: var(--paper);
        border: 1px solid var(--rule);
        border-top: 5px solid var(--red);
        border-radius: 8px;
        min-height: 260px;
        padding: 1.35rem 1.4rem;
        box-sizing: border-box;
    }
    .finding-card .source {
        color: var(--red);
        font-size: 0.76rem;
        font-weight: 760;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .finding-card h2 {
        color: var(--ink);
        font-size: 1.65rem;
        line-height: 1.15;
        margin: 0.55rem 0 0.8rem 0;
    }
    .finding-card .statement {
        color: var(--ink);
        font-size: 1.04rem;
        line-height: 1.55;
        margin: 0;
    }
    div[data-testid="stTabs"] button { font-weight: 650; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal);
        border-bottom-color: var(--teal);
    }
    div[data-testid="stImage"] img { border-radius: 8px; }
    @media (max-width: 780px) {
        .block-container { padding-top: 1rem; }
        .hero h1 { font-size: 2.2rem; }
        .hero .subtitle { font-size: 1.06rem; }
        .finding-grid { grid-template-columns: 1fr; }
        .finding-card { min-height: 0; padding: 1.15rem; }
        .big-number .number { font-size: 2.2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <div class="kicker">AI collusion research</div>
        <h1>From hidden intent to measurable restrictions</h1>
        <p class="subtitle">Because LLMs hide intent, freeze prices, and
        inherit cartels from shared history, we must simulate market
        restrictions.</p>
        <span class="badge">Verified empirical findings | Collusim simulation</span>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_observation, tab_causal = st.tabs(
    [
        "Tab 1: The Problem",
        "Tab 2: The Solution",
    ]
)


with tab_observation:
    st.markdown(
        """
        <div class="finding-grid">
            <article class="finding-card">
                <div class="source">Teammate 1 - Stasis</div>
                <h2>Price Stasis</h2>
                <p class="statement">Real LLM cartels freeze prices
                (87% hold rate), blinding action-based detectors.</p>
            </article>
            <article class="finding-card">
                <div class="source">Teammate 1 - Intent Gap</div>
                <h2>Intent vs. Action</h2>
                <p class="statement">Instructing agents to collude produces
                no reliable behavioral signal. Intent does not equal
                action.</p>
            </article>
            <article class="finding-card">
                <div class="source">Teammate 2 - History Trap</div>
                <h2>Shared History</h2>
                <p class="statement">Swapping agents fails. Newcomers read
                high-price history as proof of demand and inherit the cartel
                (Profit Gain = 0.80 vs 0.32).</p>
            </article>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if q11_baseline is not None and q11_ablation is not None:
        st.pyplot(
            starter_q11_figure(
                q11_baseline,
                q11_ablation,
            )
        )
        st.markdown(
            '<div class="explain">Starter Code Q1.1: Visible rounds are '
            "necessary. Removing history crashes prices.</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Average price: {q11_baseline['mean_price'].mean():.3f} with "
            f"100 visible rounds vs {q11_ablation['mean_price'].mean():.3f} "
            f"with 0; competitive benchmark "
            f"{q11_baseline['competitive_benchmark'].iloc[0]:.3f}. Offline "
            "one-seed starter-path validation uses a deterministic local "
            "OpenAI-compatible mock, not a hosted LLM."
        )
    else:
        st.info(
            "Starter Q1.1 artifacts are missing. Run "
            "`python tools/run_starter_validation.py`."
        )


with tab_causal:
    figure, _ = frontier_figure(signal_frontier)
    eligible = objective_ranking.loc[
        objective_ranking["detector_valid_restriction"]
    ]
    winner = eligible.iloc[0] if not eligible.empty else None
    lever_label = {
        "precision": "Noise injection",
        "delay": "Price delay",
        "visibility": "Visibility loss",
        "history_reset": "History reset",
    }
    if winner is None:
        st.error(
            "No restriction passed both the positive-control and detector "
            "evidence gates."
        )
    else:
        reduction_pct = float(winner["markup_reduction_pct"])
        cmi_reduction_pct = float(winner["cmi_reduction_pct"])
        welfare_change_pct = float(winner["welfare_change_pct"])
        if welfare_change_pct > 0:
            welfare_phrase = f"{welfare_change_pct:.2%} welfare cost"
        else:
            welfare_phrase = f"{abs(welfare_change_pct):.2%} welfare improvement"
        win_name = {
            "precision": "noise-injection",
            "delay": "delay-based",
            "visibility": "visibility-loss",
            "history_reset": "history-reset",
        }[winner["lever"]]
        st.markdown(
            f"""
            <div class="big-number">
                <div class="number">{reduction_pct:.0%} {win_name} win</div>
                <div class="copy">{lever_label[winner["lever"]]} at strength
                {winner["strength"]:.2f}: it removes {reduction_pct:.1%} of
                collusive markup and
                {cmi_reduction_pct:.1%} of state-conditional action
                dependence, with a {welfare_phrase}. Efficiency ratio
                {winner["efficiency_ratio"]:.2f}
                (90% CI [{winner["efficiency_ci_low"]:.2f},
                {winner["efficiency_ci_high"]:.2f}]).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if q15_baseline is not None and q15_ablation is not None:
        q15_metrics = [
            "average_price",
            "seller_profit",
            "trade_volume",
            "buyer_welfare",
        ]
        q15_table = pd.DataFrame(
            {
                "Baseline (messages ON)": q15_baseline.iloc[0][q15_metrics],
                "Ablation (10-character cap)": q15_ablation.iloc[0][q15_metrics],
            }
        )
        q15_table.index = [
            "Average Price",
            "Seller Profit",
            "Trade Volume",
            "Buyer Welfare",
        ]
        st.dataframe(q15_table, width="stretch")
        baseline_volume = float(q15_baseline.iloc[0]["trade_volume"])
        ablation_volume = float(q15_ablation.iloc[0]["trade_volume"])
        volume_cost_pct = (
            max(0.0, baseline_volume - ablation_volume) / baseline_volume
            if baseline_volume > 0
            else 0.0
        )
        st.markdown(
            '<div class="explain">Starter Code Q1.5: Message caps reduce '
            "coordination, but destroy buyer welfare/trade volume. Collusim\'s "
            "Precision Loss avoids this trade-off.</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Trade volume falls {volume_cost_pct:.1%}. "
            "Offline one-seed official starter validation using a deterministic "
            "local OpenAI-compatible mock, not hosted GPT or Claude."
        )
    else:
        st.info(
            "Starter Q1.5 artifacts are missing. Run "
            "`python tools/run_starter_validation.py`."
        )
    history_row = objective_ranking.loc[
        objective_ranking["lever"] == "history_reset"
    ].iloc[0]
    reset_markup = float(history_row["markup_reduction_pct"])
    reset_direction = "reduction" if reset_markup >= 0 else "increase"
    st.caption(
        "One-time history reset: "
        f"{abs(reset_markup):.1%} markup {reset_direction}, "
        f"{float(history_row['cmi_reduction_pct']):.1%} CMI reduction, "
        f"rank {int(history_row['rank'])}. It does not beat continuous "
        "noise injection."
    )
    st.markdown(
        '<div class="explain">History Reset (Teammate 2) breaks collusion '
        "but requires a disruptive market pause. Precision Loss achieves the "
        "same without interrupting trading.</div>",
        unsafe_allow_html=True,
    )
    st.pyplot(figure)
    st.pyplot(
        two_metric_figure(
            frontier,
            baseline_markup=baseline_markup,
            baseline_cmi=baseline_cmi,
        )
    )
    control = objective_controls.set_index("control")
    rct_control = control.loc["rct_hard_ban"]
    scripted_control = control.loc["scripted_benchmark"]
    st.caption(
        "Positive controls passed: calibrated agents exceed the hard ban by "
        f"{float(rct_control['markup_gap']):.3f} mean markup and "
        f"{float(rct_control['cmi_gap_bits']):.3f} CMI bits; scripted "
        f"colluders exceed competitive agents by "
        f"{float(scripted_control['markup_gap']):.3f} markup and "
        f"{float(scripted_control['cmi_gap_bits']):.3f} CMI bits."
    )
    st.caption(
        "CMI can fall because active signaling stopped or because prices "
        "froze. Read it together with markup: a freeze is not a successful "
        "restriction."
    )
    st.markdown(
        f'<div class="explain">Using Collusim, our calibrated simulator, we ran '
        "randomized signal-degradation ablations. The ranking requires a "
        "positive lower confidence bound for both the price-level reduction "
        "and the CMI reduction. The selected restriction is the highest "
        "welfare-efficiency option among rows that pass those gates.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Restriction ranking")
    ranking_display = objective_ranking[
        [
            "rank",
            "lever",
            "strength",
            "mean_price",
            "markup_reduction_pct",
            "cmi_reduction_pct",
            "welfare_change_pct",
            "efficiency_ratio",
            "detector_interpretation",
        ]
    ].copy()
    ranking_display.insert(
        5,
        "detector",
        [
            f"{cmi:.3f} bits | z={z:.2f} | flag {flag:.0%}"
            for cmi, z, flag in zip(
                objective_ranking["cmi_bits"],
                objective_ranking["cmi_z"],
                objective_ranking["cmi_flagged_share"],
            )
        ],
    )
    ranking_display["markup_reduction_pct"] = ranking_display[
        "markup_reduction_pct"
    ].map(lambda value: f"{value:.1%}")
    ranking_display["cmi_reduction_pct"] = ranking_display[
        "cmi_reduction_pct"
    ].map(lambda value: f"{value:.1%}")
    ranking_display["welfare_change_pct"] = ranking_display[
        "welfare_change_pct"
    ].map(lambda value: f"{value:+.2%}")
    ranking_display["efficiency_ratio"] = ranking_display[
        "efficiency_ratio"
    ].map(lambda value: f"{value:.2f}")
    st.dataframe(
        ranking_display,
        hide_index=True,
        width="stretch",
    )
    with st.expander("Positive controls"):
        st.dataframe(objective_controls, hide_index=True, width="stretch")
        detector_spec = objective_summary["detector_spec"]
        st.json(detector_spec)
    with st.expander("RCT frontier data"):
        st.dataframe(
            frontier[
                [
                    "lever",
                    "strength",
                    "mean_price",
                    "mean_markup",
                    "cmi_bits",
                    "cmi_z",
                    "cmi_reduction",
                    "restriction_status",
                    "efficiency_ratio",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "Causal claims are limited to Collusim. The simulator is calibrated "
        "from observational LLM behavior and is not a real-market experiment."
    )
