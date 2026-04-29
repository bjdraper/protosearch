# CLAUDE.md — protosearch

Machine-readable project summary for Claude Code and future sessions.

## Project Overview

**protosearch** is a generalised protein family survey pipeline.
Accepts any PFAM domain ID or seed FASTA and surveys a user-supplied proteome.

Pipeline:
1. HMMER domain prefilter (PF-ID provided by user)
2. ESM2-650M embedding of filtered hits
3. Leiden clustering + t-SNE visualisation
4. Per-cluster MAFFT alignment + IQ-TREE2 ML phylogeny
5. IQ-TREE2 ancestral state reconstruction
6. Cluster labeling via reference probe nearest-neighbour assignment

## GitHub Repository

- **Repo:** https://github.com/bjdraper/protosearch
- **Visibility:** Public
- **Local path:** `/Users/bendraper/project02/protosearch/`
- **Branch:** main

## Directory Layout

```
protosearch/
├── acyltransferase/          ← Python package (Wave 2 rename → protosearch/ pending)
│   ├── config.py             ← generic config loader
│   ├── utils.py              ← FASTA I/O, UniProt fetch, length filter, dedup
│   ├── search.py             ← HMMER runner, FAISS build/query, HMM download
│   ├── embed.py              ← ESM2-650M embedding
│   ├── cluster.py            ← PCA, Leiden, t-SNE
│   ├── tree.py               ← MAFFT, FastTree, IQ-TREE2
│   ├── asr.py                ← IQ-TREE2 .state parser, consensus, variable positions
│   └── visualize.py          ← t-SNE plots, sequence logos, ancestral tables
├── notebooks/
│   ├── protein_family_survey.ipynb   ← generic Colab template (fill in probes + PFAM)
│   └── agpat_crustacea.ipynb         ← AGPAT worked example (reference copy)
├── inputs/
│   ├── seed_sequences/       ← user puts .faa/.fasta here (gitignored)
│   └── family_models/        ← user puts .hmm here (gitignored)
├── outputs/                  ← runtime outputs (gitignored)
├── examples/
│   └── acyltransferase/
│       └── config_example.yaml   ← AGPAT example config (no alphafold_targets)
├── docs/
│   └── protosearch_guide.html
├── README.md
├── requirements.txt
├── environment.colab.yml
└── setup.py
```

## Forbidden Features (do not add to this repo)

- Structure prediction (AlphaFold, ESMFold)
- PyMOL workflows
- iTOL export
- Family-specific bespoke subclustering
- HTML report generation beyond basic summary
- Any AGPAT-specific hardcoded values

Archived versions of `structure.py` and `structure_viz.py` are in
`../archive/legacy_scripts/` if needed for reference.

## Python Package

`protosearch/` — the importable package. Import as:
```python
from protosearch import embed, search, cluster, tree, asr, visualize, utils
from protosearch import load_config, Config
```

## Environment (local dev)

```
conda env: agpat_tree (Python 3.11)
Platform: macOS Apple Silicon (arm64)
Key versions: numpy=2.4.3, pandas=3.0.2, scikit-learn=1.8.0,
              biopython=1.87, fair-esm=2.0.0, faiss-cpu=1.13.2,
              torch=2.11.0, transformers=5.6.2, leidenalg=0.11.0
```

## AGPAT Worked Example

The AGPAT acyltransferase analysis is documented in `../agpat/`. It is not part of this repo.
The pre-filled notebook at `notebooks/agpat_crustacea.ipynb` serves as a reference copy only;
the canonical AGPAT version lives in `../agpat/notebooks/agpat_crustacea.ipynb`.
