"""PCA → FAISS KNN graph → Leiden community detection → t-SNE layout."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class ClusterResult:
    assignments:  pd.DataFrame        # protein_id, community, label, nearest_ref
    summary:      pd.DataFrame        # community, label, n, nearest_ref
    tsne_coords:  np.ndarray          # (N, 2) 2D layout
    label_colours: dict[str, str]     # label → hex colour


def run_clustering(
    embeddings:     np.ndarray,
    ids:            list[str],
    ref_embeddings: np.ndarray,
    ref_ids:        list[str],
    ref_colours:    dict[str, str],   # ref_id → hex colour
    ref_labels:     dict[str, str],   # ref_id → display label
    k_neighbors:    int   = 25,
    resolution:     float = 2.0,
    pca_dims:       int   = 50,
    tsne_perp:      float = 100,
    tsne_cache:     str | Path | None = None,
    random_state:   int   = 42,
    subset_ids:     list[str] | None  = None,
) -> ClusterResult:
    """
    Cluster protein embeddings using Leiden community detection.

    subset_ids: if given, only cluster embeddings for those IDs (sub-clustering).
    """
    import faiss
    import leidenalg, igraph

    # optionally filter to a subset
    if subset_ids is not None:
        subset_set = set(subset_ids)
        mask = np.array([pid in subset_set for pid in ids])
        embeddings = embeddings[mask]
        ids = [pid for pid, m in zip(ids, mask) if m]

    # PCA
    n_components = min(pca_dims, embeddings.shape[0] - 1, embeddings.shape[1])
    pca_emb = PCA(n_components=n_components, random_state=random_state).fit_transform(embeddings)

    # FAISS KNN graph
    k = min(k_neighbors, len(ids) - 1)
    index = faiss.IndexFlatL2(pca_emb.shape[1])
    index.add(pca_emb.astype(np.float32))
    dists, nbrs = index.search(pca_emb.astype(np.float32), k + 1)

    # build igraph
    edges, weights = [], []
    for i in range(len(ids)):
        for j_pos in range(1, k + 1):
            j = int(nbrs[i, j_pos])
            d = float(dists[i, j_pos])
            edges.append((i, j))
            weights.append(1.0 / (d + 1e-9))

    g = igraph.Graph(n=len(ids), edges=edges, directed=False)
    g.es["weight"] = weights
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution,
        seed=random_state,
    )
    communities = partition.membership

    # label each community by nearest reference probe
    ref_pca   = PCA(n_components=n_components, random_state=random_state).fit(embeddings)
    ref_embs  = ref_pca.transform(ref_embeddings.astype(np.float32)) if ref_embeddings is not None else None

    community_labels, community_nearest = {}, {}
    for comm_id in set(communities):
        member_idx = [i for i, c in enumerate(communities) if c == comm_id]
        member_pca = pca_emb[member_idx]
        if ref_embs is not None:
            dists_to_ref = np.linalg.norm(
                member_pca[:, np.newaxis, :] - ref_embs[np.newaxis, :, :], axis=2
            ).mean(axis=0)
            best = int(dists_to_ref.argmin())
            nearest_ref = ref_ids[best]
        else:
            nearest_ref = "unknown"
        community_labels[comm_id]  = ref_labels.get(nearest_ref, nearest_ref)
        community_nearest[comm_id] = nearest_ref

    # t-SNE
    tsne_coords = _tsne(pca_emb, perplexity=tsne_perp,
                        cache=tsne_cache, random_state=random_state)

    # assemble DataFrames
    df = pd.DataFrame({
        "protein_id":  ids,
        "community":   communities,
        "label":       [community_labels[c] for c in communities],
        "nearest_ref": [community_nearest[c] for c in communities],
    })

    summary = (df.groupby(["community", "label", "nearest_ref"])
                 .size().reset_index(name="n")
                 .sort_values("n", ascending=False))

    # build colour map (community label → hex colour from nearest ref)
    label_colours = {}
    for comm_id in set(communities):
        lbl = community_labels[comm_id]
        nr  = community_nearest[comm_id]
        label_colours[lbl] = ref_colours.get(nr, "#888888")

    return ClusterResult(
        assignments=df,
        summary=summary,
        tsne_coords=tsne_coords,
        label_colours=label_colours,
    )


def _tsne(pca_emb: np.ndarray, perplexity: float,
          cache: str | Path | None, random_state: int) -> np.ndarray:
    if cache and Path(cache).exists():
        return np.load(cache)
    from openTSNE import TSNE
    coords = TSNE(perplexity=perplexity, random_state=random_state,
                  n_jobs=-1).fit(pca_emb)
    if cache:
        np.save(cache, coords)
    return np.array(coords)
