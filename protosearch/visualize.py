"""Matplotlib-based visualisations: t-SNE, trees, ancestral plots, sequence logos."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── t-SNE scatter ─────────────────────────────────────────────────────────────

def plot_tsne(
    coords:        np.ndarray,
    labels:        list[str],
    label_colours: dict[str, str],
    ref_coords:    Optional[np.ndarray] = None,
    ref_ids:       Optional[list[str]]  = None,
    ref_labels:    Optional[dict[str, str]] = None,
    ref_colours:   Optional[dict[str, str]] = None,
    title:         str = "Clustering",
    figsize:       tuple = (10, 8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F8F8F8")

    for label in sorted(set(labels)):
        mask = np.array([l == label for l in labels])
        col  = label_colours.get(label, "#888888")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=col, s=8, alpha=0.6, linewidths=0, label=label)

    if ref_coords is not None and ref_ids is not None:
        for i, rid in enumerate(ref_ids):
            col   = (ref_colours or {}).get(rid, "#333333")
            label = (ref_labels  or {}).get(rid, rid)
            ax.scatter(ref_coords[i, 0], ref_coords[i, 1],
                       c=col, s=120, marker="*", edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(label, ref_coords[i], fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    handles = [mpatches.Patch(color=label_colours.get(l, "#888888"), label=l)
               for l in sorted(set(labels))]
    ax.legend(handles=handles, fontsize=7, framealpha=0.8,
              loc="upper left", markerscale=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


# ── FastTree / Newick tree ────────────────────────────────────────────────────

def plot_tree(
    newick_path:   str | Path,
    tip_colours:   dict[str, str],      # tip_name (or prefix) → colour
    ref_ids:       set[str] | None = None,
    title:         str = "",
    max_tips:      int = 300,
    figsize:       tuple = (14, 20),
) -> plt.Figure:
    """Rectangular cladogram coloured by tip annotation."""
    from Bio import Phylo
    from io import StringIO

    tree = next(Phylo.parse(str(newick_path), "newick"))
    terminals = tree.get_terminals()
    if len(terminals) > max_tips:
        print(f"  Warning: {len(terminals)} tips — consider collapsing")

    fig, ax = plt.subplots(figsize=figsize)
    Phylo.draw(tree, axes=ax, do_show=False)

    # colour tips
    for text_obj in ax.texts:
        name = text_obj.get_text().strip()
        for prefix, col in tip_colours.items():
            if name.startswith(prefix) or name == prefix:
                text_obj.set_color(col)
                if ref_ids and prefix in ref_ids:
                    text_obj.set_fontweight("bold")
                break

    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Sequence logos ────────────────────────────────────────────────────────────

def plot_sequence_logos(
    prob_dict:    dict[str, np.ndarray],   # node_label → (n_sites, 20)
    var_positions: list[int],              # 0-based site indices to show
    aa_order:     list[str],
    figsize_per:  tuple = (12, 1.8),
) -> plt.Figure:
    import logomaker

    n_nodes = len(prob_dict)
    fig, axes = plt.subplots(n_nodes, 1,
                              figsize=(figsize_per[0], figsize_per[1] * n_nodes))
    if n_nodes == 1:
        axes = [axes]

    for ax, (node_label, probs) in zip(axes, prob_dict.items()):
        sub_probs = probs[var_positions]
        df = pd.DataFrame(sub_probs, columns=aa_order, index=range(len(var_positions)))
        logomaker.Logo(df, ax=ax, color_scheme="chemistry", stack_order="big_on_top")
        ax.set_title(node_label, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(var_positions)))
        ax.set_xticklabels([f"p{v+1}" for v in var_positions],
                           rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("P")

    fig.suptitle("Sequence logos at key ancestral nodes (variable positions)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── Ancestral AA table ────────────────────────────────────────────────────────

def plot_ancestral_table(
    prob_dict:     dict[str, np.ndarray],
    var_positions: list[int],
    aa_order:      list[str],
    figsize:       tuple = (14, 6),
) -> plt.Figure:
    """Heatmap: rows = ancestral nodes, columns = (site × AA), cells = probability."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import Blues

    cmap = plt.get_cmap("Blues")
    node_labels = list(prob_dict.keys())
    n_pos = len(var_positions)
    n_aa  = len(aa_order)

    fig, ax = plt.subplots(figsize=figsize)
    for row_i, node_label in enumerate(node_labels):
        probs = prob_dict[node_label]
        for col_i, pos in enumerate(var_positions):
            for aa_i, aa in enumerate(aa_order):
                p = float(probs[pos, aa_i])
                x = col_i * n_aa + aa_i
                ax.add_patch(plt.Rectangle((x, row_i), 1, 1,
                             color=cmap(p), linewidth=0))
                if p > 0.4:
                    ax.text(x + 0.5, row_i + 0.5, aa,
                            ha="center", va="center", fontsize=5, color="white")

    ax.set_xlim(0, n_pos * n_aa)
    ax.set_ylim(0, len(node_labels))
    ax.set_yticks([i + 0.5 for i in range(len(node_labels))])
    ax.set_yticklabels(node_labels, fontsize=8)
    ax.set_xticks([(i * n_aa + n_aa / 2) for i in range(n_pos)])
    ax.set_xticklabels([f"p{v+1}" for v in var_positions], rotation=45, ha="right", fontsize=7)
    ax.set_title("Ancestral AA probabilities at variable positions", fontsize=11, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label="Probability")
    fig.tight_layout()
    return fig


# ── Root confidence + diversity ───────────────────────────────────────────────

def plot_root_confidence(
    prob_dict:  dict[str, np.ndarray],
    aa_order:   list[str],
    figsize:    tuple = (12, 5),
) -> plt.Figure:
    """Bar charts: mean max probability per node + cross-node AA diversity."""
    node_labels = list(prob_dict.keys())
    mean_probs  = [float(prob_dict[n].max(axis=1).mean()) for n in node_labels]

    # pairwise Hamming between consensus sequences
    consensi = np.array([prob_dict[n].argmax(axis=1) for n in node_labels])
    n_nodes  = len(node_labels)
    diversity = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = float((consensi[i] != consensi[j]).mean())
            diversity[i, j] = diversity[j, i] = d

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colours = plt.cm.viridis(np.linspace(0.2, 0.8, n_nodes))
    ax1.barh(node_labels, mean_probs, color=colours)
    ax1.set_xlabel("Mean max probability"); ax1.set_title("Ancestral confidence")
    ax1.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xlim(0, 1)

    im = ax2.imshow(diversity, cmap="YlOrRd", vmin=0, vmax=0.5)
    ax2.set_xticks(range(n_nodes)); ax2.set_xticklabels(node_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(n_nodes)); ax2.set_yticklabels(node_labels, fontsize=7)
    ax2.set_title("Consensus sequence divergence (fraction diff. sites)")
    fig.colorbar(im, ax=ax2, fraction=0.04)

    fig.tight_layout()
    return fig
