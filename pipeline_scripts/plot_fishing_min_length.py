"""Stacked bars of the share of fishing, near-stationary and transit positions for each minimum event length n.

Uses the state probabilities from the interpolated run (fit_hmm.R), as in hmm_pimin_sweep_interp.png:
a position counts toward a state when that state's probability is >= PI_MIN. With PI_MIN above 0.5,
at most one state can qualify; positions where none does are shown as "Below Pi_min".
For each n, applied within each segment to the fishing labels:
  1. Gap fill: a single non-fishing fix between two fishing fixes becomes fishing (as fishing_events.R).
  2. Minimum length: a fishing event must have at least n consecutive fixes; n = 1 keeps every event.
     Fixes from shorter events go to "Below Pi_min" (no other state reaches PI_MIN there).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATES_CSV = PROJECT_ROOT / "pipeline_outputs" / "hmm_results" / "hmm_states.csv"
OUTPUT_PNG = PROJECT_ROOT / "result_figures" / "fishing_min_length_stacked.png"
OUTPUT_CSV = PROJECT_ROOT / "pipeline_outputs" / "hmm_results" / "fishing_min_length_shares.csv"

N_VALUES = [1, 2, 3, 4, 5, 6]   # minimum fixes per fishing event
PI_MIN = 0.75                   # posterior threshold for a position to count toward a state
MAX_GAP_FIXES = 1               # non-fishing gaps of this many fixes or fewer are filled
STEP_MINUTES = 5

BELOW = "Below Pi_min"
ROLES = ["Fishing", "Near-stationary", "Transit", BELOW]
ROLE_COLORS = {"Fishing": "#2a78d6", "Near-stationary": "#eb6834", "Transit": "#1baf7a",  # as in other figures
               BELOW: "#c9c9c4"}
LABEL_TEXT_COLORS = {"Fishing": "white", "Near-stationary": "black", "Transit": "black", BELOW: "black"}


# ─────────────────────────────────────────────
#  EVENT RULES
# ─────────────────────────────────────────────

def run_starts(values: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """True where a new run starts: the value changes or a new segment begins."""
    return np.r_[True, (values[1:] != values[:-1]) | (segment[1:] != segment[:-1])]


def fill_gaps(fishing: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Fill short non-fishing runs that have fishing on both sides within the same segment."""
    starts = run_starts(fishing, segment)
    run_id = np.cumsum(starts) - 1
    run_value = fishing[starts]
    run_segment = segment[starts]
    run_length = np.bincount(run_id)

    # A run is filled if it is short, not fishing, and has fishing runs before and after it in its segment
    prev_ok = np.r_[False, run_value[:-1] & (run_segment[:-1] == run_segment[1:])]
    next_ok = np.r_[run_value[1:] & (run_segment[1:] == run_segment[:-1]), False]
    fill = ~run_value & (run_length <= MAX_GAP_FIXES) & prev_ok & next_ok
    return fishing | fill[run_id]


def drop_short(fishing: np.ndarray, segment: np.ndarray, min_fixes: int) -> np.ndarray:
    """Relabel fishing runs shorter than min_fixes as non-fishing."""
    starts = run_starts(fishing, segment)
    run_id = np.cumsum(starts) - 1
    run_length = np.bincount(run_id)
    return fishing & (run_length[run_id] >= min_fixes)


# ─────────────────────────────────────────────
#  SHARES PER N
# ─────────────────────────────────────────────

def load_states() -> tuple[pd.DataFrame, dict[str, int]]:
    """Read the HMM output and find which state number plays each role."""
    states = pd.read_csv(STATES_CSV).sort_values(["ID", "time"]).reset_index(drop=True)
    # Same ranking rule as fit_hmm.R: slowest, middle and fastest mean step length
    mean_step = states.groupby("viterbi_state")["step_km"].mean().sort_values()
    state_of_role = dict(zip(["Near-stationary", "Fishing", "Transit"], mean_step.index))
    return states, state_of_role


def role_labels(states: pd.DataFrame, state_of_role: dict[str, int]) -> dict[int, np.ndarray]:
    """Role of every position for each n; states must be sorted by segment and time."""
    segment = states["ID"].to_numpy()
    # Positions where each state's probability reaches PI_MIN
    confident = {role: states[f"prob_state{state}"].to_numpy() >= PI_MIN for role, state in state_of_role.items()}
    filled = fill_gaps(confident["Fishing"], segment)

    # Non-fishing label when fishing is not kept: a confident other state, otherwise below threshold
    other = np.where(confident["Near-stationary"], "Near-stationary",
                     np.where(confident["Transit"], "Transit", BELOW))
    return {n: np.where(drop_short(filled, segment, n), "Fishing", other) for n in N_VALUES}


def role_shares(states: pd.DataFrame, state_of_role: dict[str, int]) -> pd.DataFrame:
    """Percent of positions in each role, for every n."""
    segment = states["ID"].to_numpy()
    rows = []
    for n, role in role_labels(states, state_of_role).items():
        fishing = role == "Fishing"
        counts = pd.Series(role).value_counts().reindex(ROLES, fill_value=0)
        n_events = int((run_starts(fishing, segment) & fishing).sum())
        rows.append({"n": n, **(counts / len(role) * 100).to_dict(), "fishing_events": n_events,
                     "fishing_hours": counts["Fishing"] * STEP_MINUTES / 60})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────

def plot_shares(shares: pd.DataFrame, n_positions: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(shares))
    bottom = np.zeros(len(shares))
    for role in ROLES:
        values = shares[role].to_numpy()
        # White edges give a thin gap between stacked segments
        ax.bar(x, values, bottom=bottom, width=0.6, color=ROLE_COLORS[role], label=role,
               edgecolor="white", linewidth=1.5)
        for xi, b, v in zip(x, bottom, values):
            ax.text(xi, b + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                    color=LABEL_TEXT_COLORS[role])
        bottom += values

    ax.set_xticks(x, [f"n = {n}" for n in shares["n"]])
    ax.set_xlabel("Minimum fixes per fishing event (each fix = 5 min)")
    ax.set_ylabel(f"Share of positions at or above Pi_min = {PI_MIN} (%)")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"HMM state shares by minimum fishing event length\n"
                 f"State probability >= {PI_MIN}, interpolated run, single-fix gaps filled "
                 f"(n = {n_positions:,} positions)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")


def main() -> None:
    states, state_of_role = load_states()
    print(f"State numbers by role: {state_of_role}")
    shares = role_shares(states, state_of_role)
    shares.to_csv(OUTPUT_CSV, index=False)
    print(shares.round(2).to_string(index=False))
    print(f"Saved {OUTPUT_CSV}")
    plot_shares(shares, len(states))


if __name__ == "__main__":
    main()
