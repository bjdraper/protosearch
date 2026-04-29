"""IQ-TREE2 ancestral state reconstruction: node mapping, .state parsing, consensus sequences."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


# ── Node mapping ──────────────────────────────────────────────────────────────

def map_iqtree_nodes(treefile: str | Path):
    """
    Load IQ-TREE2 tree and assign _iqtree_name attributes by post-order traversal.
    Returns (ete3.Tree, dict[node_obj → iqtree_name]).
    IQ-TREE numbers internal nodes post-order starting at Node1.
    """
    import ete3
    tree = ete3.Tree(str(treefile), format=1)
    node_map = {}
    counter  = [1]
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            name = f"Node{counter[0]}"
            node.add_feature("_iqtree_name", name)
            node_map[name] = node
            counter[0] += 1
    return tree, node_map


def find_key_nodes(
    tree,
    node_map:       dict,
    assignments_df: pd.DataFrame,
    ref_ids:        set[str],
    subcluster_col: str = "label",
) -> pd.DataFrame:
    """
    Identify key ancestral nodes:
      - root
      - crown (MRCA of all non-reference leaves)
      - one MRCA per sub-cluster label (excluding reference probes)

    Returns DataFrame with columns: node_label, iqtree_node, bootstrap, n_descendants.
    """
    import ete3

    # crown = MRCA of all non-reference leaves
    non_ref_leaves = [n for n in tree.get_leaves()
                      if not any(n.name.startswith(r) for r in ref_ids)]

    rows = []

    # root
    root = tree.get_tree_root()
    root_name = getattr(root, "_iqtree_name", "Node_root")
    rows.append({"node_label": "anc_root", "iqtree_node": root_name,
                 "bootstrap": None, "n_descendants": len(tree.get_leaves())})

    # crown
    if len(non_ref_leaves) > 1:
        crown = tree.get_common_ancestor(non_ref_leaves)
        crown_name = getattr(crown, "_iqtree_name", "Node_crown")
        rows.append({"node_label": "anc_crown", "iqtree_node": crown_name,
                     "bootstrap": getattr(crown, "support", None),
                     "n_descendants": len(crown.get_leaves())})

    # per sub-cluster MRCA
    if assignments_df is not None and subcluster_col in assignments_df.columns:
        for label in assignments_df[subcluster_col].unique():
            members  = set(assignments_df.loc[assignments_df[subcluster_col] == label, "protein_id"])
            sc_leaves = [n for n in tree.get_leaves() if n.name in members]
            if len(sc_leaves) < 2:
                continue
            mrca      = tree.get_common_ancestor(sc_leaves)
            mrca_name = getattr(mrca, "_iqtree_name", f"Node_{label}")
            slug      = label.lower().replace(" ", "_").replace("[","").replace("]","")
            rows.append({"node_label": f"anc_{slug}_mrca", "iqtree_node": mrca_name,
                         "bootstrap": getattr(mrca, "support", None),
                         "n_descendants": len(mrca.get_leaves())})

    return pd.DataFrame(rows)


# ── .state file parsing ───────────────────────────────────────────────────────

def parse_state_file(
    state_file:   str | Path,
    target_nodes: set[str] | None = None,
    aa_order:     list[str] = AA_ORDER,
    chunksize:    int = 100_000,
) -> dict[str, np.ndarray]:
    """
    Parse IQ-TREE2 .state file.
    target_nodes: set of node names to extract; None = all nodes in file.
    Returns {node_name: array shape (n_sites, 20)}.
    """
    rename = {f"p_{a}": a for a in aa_order}
    node_data: dict[str, list] = {}

    for chunk in pd.read_csv(state_file, sep="\t", comment="#", chunksize=chunksize):
        chunk = chunk.rename(columns=rename)
        if target_nodes is not None:
            chunk = chunk[chunk["Node"].isin(target_nodes)]
        if chunk.empty:
            continue
        for node, grp in chunk.groupby("Node"):
            probs = grp[aa_order].values.astype(np.float32)
            if node not in node_data:
                node_data[node] = []
            node_data[node].append(probs)

    return {node: np.vstack(chunks) for node, chunks in node_data.items() if chunks}


# ── Consensus sequences + variable positions ──────────────────────────────────

def consensus_sequence(probs: np.ndarray, aa_order: list[str] = AA_ORDER) -> str:
    """Most probable amino acid at each site."""
    return "".join(aa_order[i] for i in probs.argmax(axis=1))


def variable_positions(
    prob_dict: dict[str, np.ndarray],
    top_n:     int = 20,
    aa_order:  list[str] = AA_ORDER,
) -> list[int]:
    """
    Select the top_n alignment positions with highest cross-node AA variability.
    Returns 0-based site indices.
    """
    # consensus at each node; Hamming diversity across nodes
    consensi = np.array([probs.argmax(axis=1) for probs in prob_dict.values()])
    # fraction of nodes that agree on the most-common AA at each site
    n_nodes, n_sites = consensi.shape
    diversity = np.array([
        1 - np.bincount(consensi[:, s], minlength=20).max() / n_nodes
        for s in range(n_sites)
    ])
    return list(np.argsort(diversity)[::-1][:top_n])
