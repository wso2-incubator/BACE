# src/coevolution/analysis/lineage.py

import networkx as nx
import pandas as pd
from loguru import logger


def build_lineage_graph(ind_df: pd.DataFrame) -> nx.DiGraph | None:
    """
    Builds and returns a NetworkX graph of the full lineage.

    Handles the 'ParentDict' structure where parents are grouped by type
    (e.g., {'code': ['C1'], 'test': ['T1']}), allowing for cross-species
    lineage tracking (Code <-> Test).

    Args:
        ind_df: DataFrame containing individual records with a 'parents' column
                containing dictionaries.

    Returns:
        nx.DiGraph or None if input is empty.
    """
    if ind_df.empty:
        logger.warning("Individual DataFrame is empty. Skipping lineage graph.")
        return None

    logger.info("Building Lineage Graph...")
    G = nx.DiGraph()

    for _, row in ind_df.iterrows():
        child_id = row["id"]

        # 1. Add the Child Node
        # We use .get() to safely handle potential missing columns in the log
        G.add_node(
            child_id,
            type=row.get("type", "Unknown"),
            status=row.get("status", "Unknown"),
            prob=row.get("probability", 0.0),
            gen_born=row.get("generation_born", -1),
            op=row.get("creation_op", "Unknown"),
        )

        # 2. Extract Parents
        # The 'parents' column contains a dict: {'code': ['id1'], 'test': ['id2']}
        parents_data = row.get("parents", {})

        # Handle cases where parents might be NaN/None (e.g. generation 0)
        if not isinstance(parents_data, dict):
            continue

        # 3. Iterate over Parent Types (Code parents, Test parents)
        for parent_type, parent_ids in parents_data.items():
            if not isinstance(parent_ids, list):
                continue

            for parent_id in parent_ids:
                # Ensure parent node exists (phantom parents from previous runs/generations)
                if not G.has_node(parent_id):
                    # We infer the type from the dictionary key ('code' or 'test')
                    G.add_node(
                        parent_id,
                        type=parent_type.capitalize(),  # 'Code' or 'Test'
                        status="Ancestor",
                    )

                # Add the edge
                # We attribute the edge with the relationship type
                G.add_edge(parent_id, child_id, parent_type=parent_type)

    logger.success(
        f"Graph created: {G.number_of_nodes()} individuals, {G.number_of_edges()} lineage connections."
    )
    return G


def get_ancestral_subgraph(G: nx.DiGraph, target_id: str) -> nx.DiGraph:
    """
    Extracts the direct lineage leading to a specific individual.

    Args:
        G: The full population graph.
        target_id: The ID of the individual to trace (e.g., the final best Code).

    Returns:
        A small subgraph containing only the target and its direct ancestors.
    """
    if target_id not in G:
        print(f"Error: Target ID {target_id} not found in graph.")
        return nx.DiGraph()

    # 1. Find all ancestors (recursive upstream search)
    ancestors = nx.ancestors(G, target_id)

    # 2. Include the target itself
    nodes_to_keep = ancestors | {target_id}

    # 3. Create the subgraph
    # .copy() ensures we don't modify the original massive graph
    subgraph = G.subgraph(nodes_to_keep).copy()

    print(
        f"Tracing lineage for {target_id}: Reduced {len(G)} nodes -> {len(subgraph)} relevant ancestors."
    )
    return subgraph
