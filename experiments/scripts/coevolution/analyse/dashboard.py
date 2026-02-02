#!/usr/bin/env python3
"""
Coevolution Dashboard for Alternating Evolution Analysis.

This module visualizes the step-by-step evolution of code and test populations
during alternating coevolution with duration=1 (code-first). It displays:
- Prior and posterior probabilities for code and test populations
- Observation matrices for each test type (private, public, unittest, differential)
- Visual indicators for frozen vs evolving populations

The dashboard uses a 5-column multi-row grid:
- Column 1: Code prior probabilities (horizontal bar chart)
- Column 2: Test prior probabilities (vertical bar chart)
- Column 3: Observation matrix (heatmap: red=fail, green=pass)
- Column 4: Code posterior probabilities (horizontal bar chart)
- Column 5: Test posterior probabilities (vertical bar chart)

Row structure:
- Row 0: Initial private test (benchmarking only, no belief updates shown)
- Rows 1-3: Generation 0 (public, unittest, differential)
- Rows 4-6: Generation 1 (public, unittest, differential)
- ...
- Final row: Final private test evaluation
"""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

from coevolution.analysis.log_parser import get_problem_ids, parse_coevolution_log

app = typer.Typer()


# ============================================================================
# Type Definitions
# ============================================================================


class EvolutionData(TypedDict):
    """Container for extracted evolution data."""

    code_progression: pd.DataFrame  # Columns: id, gen_N_prior, gen_N_posterior, ...
    test_progressions: dict[str, pd.DataFrame]  # test_type -> DataFrame
    matrices: dict[str, list[pd.DataFrame]]  # test_type -> list of matrices per gen
    num_generations: int
    is_code_evolving: list[bool]  # Per generation: True if code evolves, False if tests


# ============================================================================
# Data Extraction Functions
# ============================================================================


def extract_evolution_data(parsed_log: dict, start_with: str = "code") -> EvolutionData:
    """
    Extract evolution data from parsed logs for alternating coevolution.

    Args:
        parsed_log: ParsedLog dict from log_parser.parse_coevolution_log()
        start_with: Which population evolves first ('code' or 'test')

    Returns:
        EvolutionData with progression DataFrames and matrices
    """
    gen_stats = parsed_log.get("gen_stats", pd.DataFrame())
    individuals = parsed_log.get("individuals", pd.DataFrame())
    matrices = parsed_log.get("matrices", {})

    if gen_stats.empty:
        logger.warning("No generation stats found in parsed log")
        return EvolutionData(
            code_progression=pd.DataFrame(),
            test_progressions={},
            matrices=matrices,
            num_generations=0,
            is_code_evolving=[],
        )

    num_generations = len(gen_stats)

    # Determine which population evolves in each generation (alternating pattern)
    is_code_evolving = []
    is_code = start_with == "code"
    for _ in range(num_generations):
        is_code_evolving.append(is_code)
        is_code = not is_code

    # Debug: inspect individuals structure
    if not individuals.empty:
        logger.debug(f"Individuals DataFrame columns: {individuals.columns.tolist()}")
        logger.debug(
            f"Unique 'type' values: {individuals['type'].unique().tolist() if 'type' in individuals.columns else 'N/A'}"
        )
        logger.debug(f"Total individuals: {len(individuals)}")
        # Check if there's a test_type or suite_type column
        for col in individuals.columns:
            if "test" in col.lower() or "suite" in col.lower():
                logger.debug(
                    f"Found column '{col}': unique values = {individuals[col].unique().tolist()}"
                )

    # Extract code progression
    code_progression = _extract_code_progression(gen_stats, individuals)

    # Extract test progressions for each test type
    test_progressions = {}
    for test_type in ["public", "unittest", "differential"]:
        test_prog = _extract_test_progression(gen_stats, individuals, test_type)
        if not test_prog.empty:
            test_progressions[test_type] = test_prog

    return EvolutionData(
        code_progression=code_progression,
        test_progressions=test_progressions,
        matrices=matrices,
        num_generations=num_generations,
        is_code_evolving=is_code_evolving,
    )


def _create_events_df(individuals: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten lifecycle_events from individuals DataFrame.

    Each individual has a lifecycle_events list containing dicts with:
    - event: 'created' or 'probability_updated'
    - probability: float value
    - generation: int (which generation this event occurred in)

    Returns a flattened DataFrame with columns: id, generation, event, probability
    """
    if individuals.empty or "lifecycle_events" not in individuals.columns:
        return pd.DataFrame()

    # Store original index to preserve event order
    individuals_with_idx = individuals.reset_index(drop=True)
    individuals_with_idx["original_index"] = individuals_with_idx.index

    # Explode lifecycle_events list into separate rows
    events_exploded = individuals_with_idx.explode("lifecycle_events")

    # Filter out None/NaN values
    events_exploded = events_exploded[events_exploded["lifecycle_events"].notna()]

    if events_exploded.empty:
        return pd.DataFrame()

    # Convert each dict to columns
    events_series = events_exploded["lifecycle_events"].apply(pd.Series)

    if events_series.empty:
        logger.warning("lifecycle_events resulted in empty series")
        return pd.DataFrame()

    # Debug: show sample of details structure
    if "details" in events_series.columns:
        sample_details = (
            events_series["details"].iloc[0] if len(events_series) > 0 else None
        )
        logger.debug(f"Sample details structure: {sample_details}")

    # Extract probability from details dict if it exists there
    if (
        "details" in events_series.columns
        and "probability" not in events_series.columns
    ):
        # Probability is nested in details
        events_series["probability"] = events_series["details"].apply(
            lambda x: x.get("probability") if isinstance(x, dict) else None
        )

    # Check if probability column exists now
    if "probability" not in events_series.columns:
        logger.warning(
            f"lifecycle_events missing 'probability' column even after extraction. "
            f"Available columns: {events_series.columns.tolist()}"
        )
        return pd.DataFrame()

    logger.debug(
        f"Extracted probability column, sample values: {events_series['probability'].head().tolist()}"
    )

    # Combine with individual id
    events_df = pd.concat(
        [
            events_exploded[["id", "original_index"]].reset_index(drop=True),
            events_series.reset_index(drop=True),
        ],
        axis=1,
    )

    return events_df


def _extract_code_progression(
    gen_stats: pd.DataFrame, individuals: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract code probability progression across generations from lifecycle events.

    Creates a DataFrame with columns: id, generation, before_update,
    after_public_update, after_unittest_update, after_differential_update
    """
    if individuals.empty:
        logger.warning("No individual data available for code progression")
        return pd.DataFrame()

    # Filter for code individuals
    code_inds = individuals[individuals["type"] == "code"].copy()

    if code_inds.empty:
        logger.warning("No code individuals found")
        return pd.DataFrame()

    # Create events dataframe from lifecycle_events
    events_df = _create_events_df(code_inds)

    if events_df.empty:
        logger.warning("No lifecycle events found for code individuals")
        return pd.DataFrame()

    # Get 'created' probabilities (this is before_update for generation_born)
    created_df = events_df[events_df["event"] == "created"][
        ["id", "generation", "probability"]
    ].rename(columns={"probability": "before_update"})

    # Get 'probability_updated' events and sort by order
    prob_updates = events_df[events_df["event"] == "probability_updated"].copy()

    if prob_updates.empty:
        # Just return created probabilities
        return created_df.reset_index(drop=True)

    # Extract test_type from details dict
    prob_updates["test_type"] = prob_updates["details"].apply(
        lambda x: x.get("test_type") if isinstance(x, dict) else None
    )

    # Sort to preserve event order within each generation
    prob_updates = prob_updates.sort_values(by=["id", "generation", "original_index"])

    # Debug: Show raw probability updates for first individual in gen 0
    if not prob_updates.empty:
        first_id = prob_updates["id"].iloc[0]
        gen0_updates = prob_updates[
            (prob_updates["id"] == first_id) & (prob_updates["generation"] == 0)
        ]
        if not gen0_updates.empty:
            logger.debug(
                f"Gen 0 probability updates for {first_id}: "
                f"{gen0_updates[['test_type', 'probability', 'original_index']].to_dict('records')}"
            )

    # Map test_type to column names
    # NOTE: In gen 0 (bootstrap), unittest appears TWICE because differential tests don't exist yet
    # In gen 1+, pattern is: public → unittest → differential
    # We use 'last' aggregation to take the final value when a test_type appears multiple times
    test_type_to_col = {
        "public": "after_public_update",
        "unittest": "after_unittest_update",
        "differential": "after_differential_update",
    }
    prob_updates["stage"] = prob_updates["test_type"].map(test_type_to_col)

    # Pivot to wide format, using 'last' to take the final update if test_type appears multiple times
    pivot_df = prob_updates.pivot_table(
        index=["id", "generation"],
        columns="stage",
        values="probability",
        aggfunc="last",  # Take LAST occurrence (handles unittest appearing 2x in gen 0)
    ).reset_index()

    # Merge created and updated probabilities
    progression_df = pd.merge(
        created_df, pivot_df, on=["id", "generation"], how="outer"
    )

    # CRITICAL: Create complete skeleton for all individuals across all generations
    # Each individual needs a row for EVERY generation from birth to max_gen
    # This is essential because code gets belief updates EVERY generation (even when frozen)
    max_gen = int(progression_df["generation"].max())

    all_rows = []
    for ind_id, ind_data in progression_df.groupby("id"):
        birth_gen = int(ind_data["generation"].min())
        # Create complete index from birth to max_gen
        complete_gens = pd.DataFrame(
            {"id": ind_id, "generation": range(birth_gen, max_gen + 1)}
        )
        # Merge with actual data
        ind_complete = pd.merge(
            complete_gens, ind_data, on=["id", "generation"], how="left"
        )
        all_rows.append(ind_complete)

    progression_df = pd.concat(all_rows, ignore_index=True)
    progression_df = progression_df.sort_values(by=["id", "generation"])

    # CRITICAL: For survivors in new generations, before_update must be the
    # previous generation's final state (after_differential_update)
    # This ensures probability continuity across evolution boundaries
    # Do this BEFORE forward-fill so we don't carry forward incorrect before_update values
    for ind_id, ind_data in progression_df.groupby("id"):
        birth_gen = int(ind_data["generation"].min())
        for idx, row in ind_data.iterrows():
            gen = int(row["generation"])
            if gen > birth_gen:  # Survivor in a new generation
                # before_update = previous generation's final state
                prev_gen_data = ind_data[ind_data["generation"] == gen - 1]
                if (
                    not prev_gen_data.empty
                    and "after_differential_update" in prev_gen_data.columns
                ):
                    prev_final_prob = prev_gen_data["after_differential_update"].iloc[0]
                    if pd.notna(prev_final_prob):
                        progression_df.at[idx, "before_update"] = prev_final_prob

    # Forward-fill all other columns from previous generation as a safety net
    # With explicit zero-change logs, most values will be populated, but forward-fill
    # handles edge cases and ensures no NaN values
    # Note: before_update is already set correctly for survivors, so forward-fill won't overwrite
    for col in [
        "after_public_update",
        "after_unittest_update",
        "after_differential_update",
    ]:
        if col in progression_df.columns:
            progression_df[col] = progression_df.groupby("id")[col].ffill()

    return progression_df.reset_index(drop=True)


def _extract_test_progression(
    gen_stats: pd.DataFrame, individuals: pd.DataFrame, test_type: str
) -> pd.DataFrame:
    """
    Extract test probability progression for a specific test type across generations.

    Creates a DataFrame with columns: id, generation, before_update, after_update
    """
    if individuals.empty:
        return pd.DataFrame()

    # Filter for test individuals by test_type (type column contains: 'unittest', 'differential', 'public')
    test_inds = individuals[individuals["type"] == test_type].copy()

    if test_inds.empty:
        logger.debug(
            f"No {test_type} test individuals found (this is OK if {test_type} tests don't evolve)"
        )
        return pd.DataFrame()

    logger.debug(
        f"Found {len(test_inds)} {test_type} test individuals: {test_inds['id'].tolist()}"
    )

    # Create events dataframe from lifecycle_events
    events_df = _create_events_df(test_inds)

    if events_df.empty:
        logger.warning(f"No lifecycle events found for {test_type} test individuals")
        return pd.DataFrame()

    logger.debug(
        f"Created events_df for {test_type} with {len(events_df)} rows, "
        f"event types: {events_df['event'].unique().tolist() if 'event' in events_df.columns else 'N/A'}"
    )

    # Get 'created' probabilities
    created_df = events_df[events_df["event"] == "created"][
        ["id", "generation", "probability"]
    ].rename(columns={"probability": "before_update"})

    # Get 'probability_updated' events (tests only have one update per generation)
    prob_updates = events_df[events_df["event"] == "probability_updated"].copy()

    if prob_updates.empty:
        # Just return created probabilities
        return created_df.reset_index(drop=True)

    # Sort to preserve event order
    prob_updates = prob_updates.sort_values(by=["id", "generation", "original_index"])

    # Take the first (and typically only) update per generation
    df_update = (
        prob_updates.groupby(["id", "generation"])
        .first()
        .reset_index()[["id", "generation", "probability"]]
        .rename(columns={"probability": "after_update"})
    )

    # Merge created and updated probabilities
    progression_df = pd.merge(
        created_df, df_update, on=["id", "generation"], how="outer"
    )

    # CRITICAL: Create complete skeleton for all individuals across all generations
    max_gen = int(progression_df["generation"].max())

    all_rows = []
    for ind_id, ind_data in progression_df.groupby("id"):
        birth_gen = int(ind_data["generation"].min())
        # Create complete index from birth to max_gen
        complete_gens = pd.DataFrame(
            {"id": ind_id, "generation": range(birth_gen, max_gen + 1)}
        )
        # Merge with actual data
        ind_complete = pd.merge(
            complete_gens, ind_data, on=["id", "generation"], how="left"
        )
        all_rows.append(ind_complete)

    progression_df = pd.concat(all_rows, ignore_index=True)
    progression_df = progression_df.sort_values(by=["id", "generation"])

    # For each individual, forward-fill all columns from previous generation
    for col in ["before_update", "after_update"]:
        if col in progression_df.columns:
            progression_df[col] = progression_df.groupby("id")[col].ffill()

    return progression_df.reset_index(drop=True)


# ============================================================================
# Plotting Helper Functions
# ============================================================================


def plot_code_prob_horizontal(
    prob_data: pd.DataFrame,
    prob_column: str,
    title: str,
    ax: plt.Axes,
    is_frozen: bool = False,
    aligned_rows: list[str] | None = None,
) -> None:
    """
    Plot code probabilities as horizontal bar chart.

    Args:
        prob_data: DataFrame with 'id' and probability columns
        prob_column: Column name containing probabilities to plot
        title: Subplot title
        ax: Matplotlib axes
        is_frozen: If True, use visual indicators for frozen population
        aligned_rows: Optional list of code IDs to align with matrix rows
    """
    if prob_data is None or prob_data.empty or prob_column not in prob_data.columns:
        ax.text(
            0.5,
            0.5,
            "No Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#CCCCCC",
            fontsize=11,
            style="italic",
        )
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Align data to matrix row index if provided
    if aligned_rows is not None:
        plot_data = prob_data.set_index("id").reindex(aligned_rows)
    else:
        plot_data = prob_data.set_index("id")

    # Use actual index for y-positioning (reversed so matrix row 0 is at top)
    y_positions = np.arange(len(plot_data))[
        ::-1
    ]  # Reverse to match matrix top-to-bottom

    color = "#B0E0E6"  # Pastel blue for code

    ax.barh(
        y=y_positions,
        width=plot_data[prob_column].fillna(0),
        color=color,
        edgecolor="white" if not is_frozen else "lightgray",
        linewidth=0.5,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data.index, fontsize=8)
    ax.set_title(
        title, fontsize=10, pad=5, weight="semibold" if not is_frozen else "normal"
    )
    ax.set_xlabel("Probability" if not is_frozen else "Prob (Frozen)", fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_test_prob_vertical(
    prob_data: pd.DataFrame,
    prob_column: str,
    title: str,
    ax: plt.Axes,
    is_frozen: bool = False,
    aligned_cols: list[str] | None = None,
) -> None:
    """
    Plot test probabilities as vertical bar chart.

    Args:
        prob_data: DataFrame with 'id' and probability columns
        prob_column: Column name containing probabilities to plot
        title: Subplot title
        ax: Matplotlib axes
        is_frozen: If True, use visual indicators for frozen population
        aligned_cols: Optional list of test IDs to align with matrix columns
    """
    if prob_data is None or prob_data.empty or prob_column not in prob_data.columns:
        ax.text(
            0.5,
            0.5,
            "No Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#CCCCCC",
            fontsize=11,
            style="italic",
        )
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Align data to matrix column index if provided
    if aligned_cols is not None:
        plot_data = prob_data.set_index("id").reindex(aligned_cols)
    else:
        plot_data = prob_data.set_index("id")

    # Use actual index for x-positioning
    x_positions = np.arange(len(plot_data))

    color = "#FFDAB9"  # Pastel orange for tests

    ax.bar(
        x=x_positions,
        height=plot_data[prob_column].fillna(0),
        color=color,
        edgecolor="white" if not is_frozen else "lightgray",
        linewidth=0.5,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_data.index, rotation=90, fontsize=8, ha="center")
    ax.set_title(
        title, fontsize=10, pad=5, weight="semibold" if not is_frozen else "normal"
    )
    ax.set_ylabel("Probability" if not is_frozen else "Prob (Frozen)", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_matrix_heatmap(
    matrix_df: pd.DataFrame, title: str, ax: plt.Axes, is_benchmark: bool = False
) -> None:
    """
    Plot observation matrix as heatmap (red=fail, green=pass).

    Args:
        matrix_df: DataFrame with code IDs as rows, test IDs as columns
        title: Subplot title
        ax: Matplotlib axes
        is_benchmark: If True, use muted colors for benchmark rows
    """
    if matrix_df is None or matrix_df.empty:
        ax.text(
            0.5,
            0.5,
            "No Matrix Data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#CCCCCC",
            fontsize=11,
            style="italic",
        )
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Convert to binary matrix
    mat = matrix_df.astype(int).clip(0, 1).values

    # Define colormap: 0=red (fail), 1=green (pass)
    if is_benchmark:
        # Muted colors for benchmark
        colors = ["#FFB6C1", "#98FB98"]  # Pastel red, pastel green
    else:
        colors = ["#FFB6C1", "#98FB98"]  # Pastel red, pastel green

    cmap = ListedColormap(colors)

    # Plot heatmap
    ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns, rotation=90, fontsize=7, ha="center")
    ax.set_yticklabels(matrix_df.index, fontsize=7)

    ax.set_title(
        title, fontsize=10, pad=5, weight="normal" if is_benchmark else "semibold"
    )
    ax.set_xlabel("Test IDs", fontsize=8)
    ax.set_ylabel("Code IDs", fontsize=8)
    ax.tick_params(labelsize=7)

    # Add grid
    ax.set_xticks(np.arange(len(matrix_df.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix_df.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)


def plot_placeholder(ax: plt.Axes, text: str = "Not Updated") -> None:
    """Plot a placeholder for columns that don't show data."""
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#999999",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================================
# Row Plotting Functions
# ============================================================================


def plot_private_row(
    gs: gridspec.GridSpec,
    fig: plt.Figure,
    code_prog: pd.DataFrame,
    test_prog: pd.DataFrame,
    matrix: pd.DataFrame,
    generation: int,
    label: str = "INITIAL PRIVATE",
    use_final_state: bool = False,
) -> None:
    """
    Plot initial/final private test row (benchmarking only, no belief updates).

    Columns 4-5 show placeholders since private tests don't update beliefs.

    Args:
        use_final_state: If True, shows after_differential_update (for final row)
    """
    # Create 5 subplots
    ax_code_prior = fig.add_subplot(gs[0])
    ax_test_prior = fig.add_subplot(gs[1])
    ax_matrix = fig.add_subplot(gs[2])
    ax_code_post = fig.add_subplot(gs[3])
    ax_test_post = fig.add_subplot(gs[4])

    # Add row label on the left
    ax_code_prior.set_ylabel(
        label,
        fontsize=11,
        weight="bold",
        color="#666666",
        labelpad=10,
        rotation=0,
        ha="right",
    )

    # Get aligned IDs from matrix
    code_ids = matrix.index.tolist() if not matrix.empty else None
    test_ids = matrix.columns.tolist() if not matrix.empty else None

    # Filter progression data for this generation
    code_gen_data = (
        code_prog[code_prog["generation"] == generation]
        if not code_prog.empty and "generation" in code_prog.columns
        else pd.DataFrame()
    )
    test_gen_data = (
        test_prog[test_prog["generation"] == generation]
        if not test_prog.empty and "generation" in test_prog.columns
        else pd.DataFrame()
    )

    # For final private row, use after_differential_update; for initial, use before_update
    code_prob_col = "after_differential_update" if use_final_state else "before_update"

    # Plot prior probabilities (final state or initial state)
    plot_code_prob_horizontal(
        code_gen_data,
        code_prob_col,
        "Code State",
        ax_code_prior,
        is_frozen=False,
        aligned_rows=code_ids,
    )

    plot_test_prob_vertical(
        test_gen_data,
        "before_update",
        "Private Test Prior",
        ax_test_prior,
        is_frozen=False,
        aligned_cols=test_ids,
    )

    # Plot observation matrix
    plot_matrix_heatmap(
        matrix, "Private Observation Matrix", ax_matrix, is_benchmark=True
    )

    # Plot placeholders for posteriors (private doesn't update beliefs)
    plot_placeholder(ax_code_post, "Benchmark Only")
    plot_placeholder(ax_test_post, "Benchmark Only")


def plot_test_evolution_row(
    gs: gridspec.GridSpec,
    fig: plt.Figure,
    code_prog: pd.DataFrame,
    test_prog: pd.DataFrame,
    matrix: pd.DataFrame,
    generation: int,
    test_type: str,
    is_code_evolving: bool,
    code_gen: int,
    test_gen: int,
) -> None:
    """
    Plot a row for public/unittest/differential test evolution.

    Args:
        gs: GridSpec for this row
        fig: Figure
        code_prog: Code progression DataFrame
        test_prog: Test progression DataFrame for this test type
        matrix: Observation matrix for this test type
        generation: Current dashboard generation number (global counter)
        test_type: 'public', 'unittest', or 'differential'
        is_code_evolving: True if code is evolving this generation
        code_gen: Current code generation number
        test_gen: Current test generation number
    """
    # Create 5 subplots
    ax_code_prior = fig.add_subplot(gs[0])
    ax_test_prior = fig.add_subplot(gs[1])
    ax_matrix = fig.add_subplot(gs[2])
    ax_code_post = fig.add_subplot(gs[3])
    ax_test_post = fig.add_subplot(gs[4])

    # Add row label with SEPARATE generation numbers for code and tests
    label = f"Code G{code_gen} / Test G{test_gen}\n{test_type.upper()}"
    evolution_status = "Code Evolves" if is_code_evolving else "Tests Evolve"
    ax_code_prior.set_ylabel(
        f"{label}\n[{evolution_status}]",
        fontsize=10,
        weight="semibold",
        color="#333333",
        labelpad=10,
        rotation=0,
        ha="right",
        va="center",
    )

    # Get aligned IDs from matrix
    code_ids = matrix.index.tolist() if not matrix.empty else None
    test_ids = matrix.columns.tolist() if not matrix.empty else None

    # Filter progression data
    # KEY: When a population is EVOLVING, we need data from TWO generations:
    #   - Prior: The PREVIOUS generation's final state (survivors carry forward)
    #   - Posterior: The CURRENT generation's final state (after evolution)
    # When a population is FROZEN, we use the same generation for both (they just get belief updates)

    if is_code_evolving:
        # Code is evolving: All three test rows test the CURRENT code generation
        # Evolution happens AFTER all test updates, creating the next generation
        # So both prior and posterior come from the same code generation
        code_prior_gen = code_gen
        code_posterior_gen = code_gen
    else:
        # Code is frozen: both from same code generation (it just gets belief updates)
        code_prior_gen = code_gen
        code_posterior_gen = code_gen

    if not is_code_evolving:
        # Tests are evolving: prior from current gen (before evolution), posterior from NEXT gen (after evolution)
        test_prior_gen = test_gen
        test_posterior_gen = test_gen + 1  # New individuals born in next generation
    else:
        # Tests are frozen: both from same test generation
        test_prior_gen = test_gen
        test_posterior_gen = test_gen

    code_prior_data = (
        code_prog[code_prog["generation"] == code_prior_gen]
        if not code_prog.empty and "generation" in code_prog.columns
        else pd.DataFrame()
    )
    code_posterior_data = (
        code_prog[code_prog["generation"] == code_posterior_gen]
        if not code_prog.empty and "generation" in code_prog.columns
        else pd.DataFrame()
    )
    test_prior_data = (
        test_prog[test_prog["generation"] == test_prior_gen]
        if not test_prog.empty and "generation" in test_prog.columns
        else pd.DataFrame()
    )
    test_posterior_data = (
        test_prog[test_prog["generation"] == test_posterior_gen]
        if not test_prog.empty and "generation" in test_prog.columns
        else pd.DataFrame()
    )

    # Determine which columns to use based on test type
    # Within a generation, each test type uses the previous type's posterior as its prior
    if is_code_evolving:
        # Code is evolving, tests are FROZEN
        # Tests already ran in the previous generation, so no new updates
        # All rows should show carried-forward probabilities (no change)
        if test_type == "public":
            # First row: show carried-forward state from previous generation
            code_prior_col = "before_update"  # Start of this generation
            code_prior_data_for_plot = code_prior_data  # From current gen (code_gen)
        elif test_type == "unittest":
            # After public: no change (tests frozen)
            code_prior_col = "after_public_update"
            code_prior_data_for_plot = code_prior_data  # Same gen (code_gen)
        else:  # differential
            # After unittest: no change (tests frozen)
            code_prior_col = "after_unittest_update"
            code_prior_data_for_plot = code_prior_data  # Same gen (code_gen)
    else:
        # Code is frozen: use within-generation chaining (all from same gen)
        code_prior_map = {
            "public": "before_update",  # Start of generation
            "unittest": "after_public_update",  # After public tests
            "differential": "after_unittest_update",  # After unittest tests
        }
        code_prior_col = code_prior_map.get(test_type, "before_update")
        code_prior_data_for_plot = code_prior_data  # All from same code gen

    code_posterior_map = {
        "public": "after_public_update",
        "unittest": "after_unittest_update",
        "differential": "after_differential_update",
    }
    code_posterior_col = code_posterior_map.get(test_type, "after_public_update")

    # Debug: Log probability values for verification with IDs
    if (
        not code_prior_data_for_plot.empty
        and code_prior_col in code_prior_data_for_plot.columns
    ):
        logger.debug(
            f"Code G{code_gen}, Test G{test_gen} {test_type}: "
            f"code_prior_gen={code_prior_gen}, code_posterior_gen={code_posterior_gen}"
        )
        logger.debug(
            f"Prior data ({code_prior_col}): "
            f"{code_prior_data_for_plot[['id', code_prior_col]].head(5).to_dict('records')}"
        )
    if (
        not code_posterior_data.empty
        and code_posterior_col in code_posterior_data.columns
    ):
        logger.debug(
            f"Posterior data ({code_posterior_col}): "
            f"{code_posterior_data[['id', code_posterior_col]].head(5).to_dict('records')}"
        )

    # Plot prior probabilities
    plot_code_prob_horizontal(
        code_prior_data_for_plot,
        code_prior_col,
        "Code Prior",
        ax_code_prior,
        is_frozen=not is_code_evolving,
        aligned_rows=code_ids,
    )

    plot_test_prob_vertical(
        test_prior_data,
        "before_update"
        if not is_code_evolving
        else "after_update",  # Use final state if tests evolved previously
        f"{test_type.title()} Prior",
        ax_test_prior,
        is_frozen=is_code_evolving,  # Tests frozen when code evolves
        aligned_cols=test_ids,
    )

    # Plot observation matrix
    plot_matrix_heatmap(matrix, f"{test_type.title()} Observation Matrix", ax_matrix)

    # Plot posterior probabilities
    plot_code_prob_horizontal(
        code_posterior_data,
        code_posterior_col,
        "Code Posterior",
        ax_code_post,
        is_frozen=not is_code_evolving,
        aligned_rows=code_ids,
    )

    # Note: Public tests don't evolve, so show as frozen
    test_frozen = is_code_evolving or test_type == "public"
    plot_test_prob_vertical(
        test_posterior_data,
        "after_update",
        f"{test_type.title()} Posterior",
        ax_test_post,
        is_frozen=test_frozen,
        aligned_cols=test_ids,
    )


# ============================================================================
# Main Dashboard Assembly
# ============================================================================


def plot_alternating_dashboard(
    evolution_data: EvolutionData,
    output_path: str | None = None,
    figsize_per_row: float = 4.0,
) -> None:
    """
    Create the full alternating coevolution dashboard.

    Args:
        evolution_data: Extracted evolution data
        output_path: Optional path to save figure
        figsize_per_row: Height per row in inches
    """
    num_gens = evolution_data["num_generations"]
    matrices = evolution_data["matrices"]

    # Calculate total rows: 1 initial private + 3 rows per gen + 1 final private
    n_rows_per_gen = 3  # public, unittest, differential
    n_total_rows = 1 + (num_gens * n_rows_per_gen) + 1

    logger.info(
        f"Creating dashboard with {n_total_rows} rows "
        f"({num_gens} generations, 3 test types per generation)"
    )

    # Create figure
    fig_height = n_total_rows * figsize_per_row
    fig = plt.figure(figsize=(20, fig_height))

    # Determine title based on which population starts evolving
    start_pop = "Code" if evolution_data["is_code_evolving"][0] else "Tests"
    fig.suptitle(
        f"Alternating Coevolution Dashboard ({start_pop}-First, Duration=1)",
        fontsize=16,
        weight="bold",
        y=0.995,
    )

    # Create outer GridSpec
    outer_gs = gridspec.GridSpec(
        n_total_rows, 1, figure=fig, hspace=0.5, top=0.98, bottom=0.02
    )

    row_idx = 0

    # Row 0: Initial private test (before any evolution, so generation=-1 conceptually)
    logger.info("Plotting initial private test row...")
    gs_init = outer_gs[row_idx].subgridspec(
        1, 5, width_ratios=[1.2, 1.2, 3, 1.2, 1.2], wspace=0.20
    )
    initial_private_matrix = matrices.get("private", [pd.DataFrame()])[0]
    # Use generation 0's 'before_update' state (initial population)
    plot_private_row(
        gs_init,
        fig,
        evolution_data["code_progression"],
        evolution_data["test_progressions"].get("public", pd.DataFrame()),
        initial_private_matrix,
        generation=0,
        label="INITIAL\nPRIVATE\n(Benchmark)",
    )
    row_idx += 1

    # Loop through generations, tracking separate generation counters for code and tests
    code_gen = 0  # Code generation counter
    test_gen = 0  # Test generation counter

    for gen in range(num_gens):
        is_code_evolving = evolution_data["is_code_evolving"][gen]
        logger.info(
            f"Plotting generation {gen} "
            f"({'Code Evolving' if is_code_evolving else 'Tests Evolving'})..."
        )

        # Get matrices for this generation
        public_matrices = matrices.get("public", [])
        unittest_matrices = matrices.get("unittest", [])
        differential_matrices = matrices.get("differential", [])

        public_matrix = (
            public_matrices[gen] if gen < len(public_matrices) else pd.DataFrame()
        )
        unittest_matrix = (
            unittest_matrices[gen] if gen < len(unittest_matrices) else pd.DataFrame()
        )
        differential_matrix = (
            differential_matrices[gen]
            if gen < len(differential_matrices)
            else pd.DataFrame()
        )

        # Row for public tests
        gs_public = outer_gs[row_idx].subgridspec(
            1, 5, width_ratios=[1.2, 1.2, 3, 1.2, 1.2], wspace=0.20
        )
        plot_test_evolution_row(
            gs_public,
            fig,
            evolution_data["code_progression"],
            evolution_data["test_progressions"].get("public", pd.DataFrame()),
            public_matrix,
            gen,
            "public",
            is_code_evolving,
            code_gen,
            test_gen,
        )
        row_idx += 1

        # Row for unittest tests
        gs_unittest = outer_gs[row_idx].subgridspec(
            1, 5, width_ratios=[1.2, 1.2, 3, 1.2, 1.2], wspace=0.20
        )
        plot_test_evolution_row(
            gs_unittest,
            fig,
            evolution_data["code_progression"],
            evolution_data["test_progressions"].get("unittest", pd.DataFrame()),
            unittest_matrix,
            gen,
            "unittest",
            is_code_evolving,
            code_gen,
            test_gen,
        )
        row_idx += 1

        # Row for differential tests
        gs_differential = outer_gs[row_idx].subgridspec(
            1, 5, width_ratios=[1.2, 1.2, 3, 1.2, 1.2], wspace=0.20
        )
        plot_test_evolution_row(
            gs_differential,
            fig,
            evolution_data["code_progression"],
            evolution_data["test_progressions"].get("differential", pd.DataFrame()),
            differential_matrix,
            gen,
            "differential",
            is_code_evolving,
            code_gen,
            test_gen,
        )
        row_idx += 1

        # Increment the appropriate generation counter after this generation completes
        if is_code_evolving:
            code_gen += 1
        else:
            test_gen += 1

    # Final row: Final private test (after all evolution, uses last generation's final state)
    logger.info("Plotting final private test row...")
    gs_final = outer_gs[row_idx].subgridspec(
        1, 5, width_ratios=[1.2, 1.2, 3, 1.2, 1.2], wspace=0.20
    )
    final_private_matrix = matrices.get("private", [pd.DataFrame()])[-1]
    # Use last generation's 'after_differential_update' state
    plot_private_row(
        gs_final,
        fig,
        evolution_data["code_progression"],
        evolution_data["test_progressions"].get("public", pd.DataFrame()),
        final_private_matrix,
        generation=num_gens - 1,  # Last generation, not num_gens
        label="FINAL\nPRIVATE\n(Benchmark)",
        use_final_state=True,
    )

    # Save or show
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.success(f"Dashboard saved to: {output_file}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


# ============================================================================
# CLI Interface
# ============================================================================


@app.command()
def main(
    run_id: str = typer.Option(..., "--run-id", "-r", help="The Run ID to analyze"),
    log_dir: str = typer.Option(
        "logs", "--log-dir", "-d", help="Directory containing log files"
    ),
    problem_id: str = typer.Option(
        None, "--problem-id", "-p", help="Problem ID (auto-select if not provided)"
    ),
    file_pattern: str = typer.Option(
        "*.log",
        "--file-pattern",
        "-f",
        help="Pattern to match log files (e.g., '*.log' or '*.zip')",
    ),
    output_path: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for dashboard image (if not provided, displays interactively)",
    ),
    start_with: str = typer.Option(
        "test",
        "--start-with",
        "-s",
        help="Which population evolves first: 'code' or 'test'",
    ),
    figsize_per_row: float = typer.Option(
        4.0, "--row-height", help="Height per row in inches"
    ),
) -> None:
    """
    Generate an alternating coevolution dashboard from log files.

    This visualizes the step-by-step evolution showing:
    - Prior and posterior probabilities for code and test populations
    - Observation matrices (red=fail, green=pass)
    - Visual indicators for frozen vs evolving populations

    Example:
        python dashboard.py --run-id abc123 --problem-id HumanEval/0 --output dashboard.png
    """
    logger.info(f"Loading logs for run_id={run_id}, problem_id={problem_id or 'auto'}")

    # Auto-select problem_id if not provided
    if problem_id is None:
        problem_ids = get_problem_ids(log_dir, file_pattern, run_id)
        if not problem_ids:
            logger.error(f"No problem_id found for run_id='{run_id}'")
            raise typer.Exit(code=1)

        problem_id = sorted(problem_ids)[0]
        logger.warning(
            f"Auto-selected problem_id='{problem_id}' from {len(problem_ids)} available"
        )

    # Parse logs
    logger.info("Parsing coevolution logs...")
    parsed_log = parse_coevolution_log(
        log_dir=log_dir,
        log_filename_pattern=file_pattern,
        target_run_id=run_id,
        target_problem_id=problem_id,
    )

    if parsed_log["gen_stats"].empty:
        logger.error("No generation data found in logs")
        raise typer.Exit(code=1)

    # Extract evolution data
    logger.info("Extracting evolution data...")
    evolution_data = extract_evolution_data(parsed_log, start_with=start_with)

    if evolution_data["num_generations"] == 0:
        logger.error("No generations found in evolution data")
        raise typer.Exit(code=1)

    logger.info(
        f"Found {evolution_data['num_generations']} generations, "
        f"{len(evolution_data['test_progressions'])} test types"
    )

    # Generate dashboard
    logger.info("Generating dashboard...")
    plot_alternating_dashboard(
        evolution_data, output_path=output_path, figsize_per_row=figsize_per_row
    )

    logger.success("Dashboard generation complete!")


if __name__ == "__main__":
    app()
