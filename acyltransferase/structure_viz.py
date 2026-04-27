"""
Structure visualisation for Colab using py3Dmol.
Replaces PyMOL for interactive in-notebook viewing.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def _compile_motifs(motif_config: dict) -> dict:
    """
    Convert config catalytic_motifs dict into compiled regex tuples.
    motif_config: {name: {pattern, colour, role}}
    Returns: {name: (compiled_re, match_length, colour)}
    """
    compiled = {}
    for name, spec in (motif_config or {}).items():
        pat = re.compile(spec["pattern"])
        # infer match length from a test match on a dummy string, or from pattern
        try:
            length = len(re.match(spec["pattern"], "A" * 20).group())
        except Exception:
            length = 4
        compiled[name] = (pat, length, spec.get("colour", "#AAAAAA"))
    return compiled


def _seq_resnums(pdb_text: str) -> tuple[str, list[int]]:
    seen, seq, rn = set(), [], []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            r = int(line[22:26]); res = line[17:20].strip()
            if r not in seen:
                seen.add(r); seq.append(AA3TO1.get(res, "X")); rn.append(r)
    return "".join(seq), rn


def find_motifs(pdb_text: str, motif_config: dict | None = None) -> dict[str, list[int]]:
    """
    Return {motif_name: [pdb_residue_numbers]} for all detected motifs.
    motif_config: from cfg.catalytic_motifs; if None, returns empty dict.
    """
    if not motif_config:
        return {}
    seq, rn   = _seq_resnums(pdb_text)
    patterns  = _compile_motifs(motif_config)
    motifs    = {}
    for name, (pat, length, _) in patterns.items():
        hit = pat.search(seq)
        if hit:
            motifs[name] = rn[hit.start():hit.start() + length]
    return motifs


def view_single(
    pdb_path:     str | Path,
    colour:       str  = "spectrum",
    style:        str  = "cartoon",
    width:        int  = 800,
    height:       int  = 600,
    motif_config: dict | None = None,
):
    """
    Display one structure in a Colab cell.
    style: 'cartoon' | 'stick' | 'sphere' | 'surface'
    colour: 'spectrum' | 'chain' | any hex colour e.g. '#FF8F00'
    """
    import py3Dmol

    pdb_text = Path(pdb_path).read_text()
    view     = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")

    style_dict = {style: {}}
    if colour == "spectrum":
        style_dict[style]["colorscheme"] = "spectral"
    elif colour == "chain":
        style_dict[style]["colorscheme"] = "chain"
    else:
        style_dict[style]["color"] = colour

    view.setStyle({}, style_dict)

    if motif_config:
        patterns  = _compile_motifs(motif_config)
        found     = find_motifs(pdb_text, motif_config)
        for mname, resi_list in found.items():
            col      = patterns[mname][2]
            resi_str = ",".join(str(r) for r in resi_list)
            view.addStyle({"resi": resi_str},
                          {"stick": {"color": col, "radius": 0.3}})
            view.addLabel(mname,
                          {"fontColor": col, "fontSize": 12, "backgroundColor": "white"},
                          {"resi": str(resi_list[0])})

    view.zoomTo(); view.spin(False)
    return view


def view_overlay(
    pdb_paths:    dict[str, str | Path],   # {label: pdb_path}
    colours:      dict[str, str],          # {label: hex_colour}
    align_ref:    Optional[str] = None,
    width:        int   = 1000,
    height:       int   = 700,
    transparency: float = 0.0,
    motif_config: dict | None = None,
) -> "py3Dmol.view":
    """
    Overlay multiple structures in one view, coloured by cluster.
    Structures are shown as cartoons; motif residues as sticks.
    Note: py3Dmol does not do structural alignment — use PyMOL for that.
    """
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)

    for label, pdb_path in pdb_paths.items():
        pdb_text = Path(pdb_path).read_text()
        col      = colours.get(label, "#888888")
        view.addModel(pdb_text, "pdb", {"vibrate": {"frames": 10, "amplitude": 1}})
        view.setStyle({"model": -1},
                      {"cartoon": {"color": col,
                                   "opacity": 1 - transparency}})

        if motif_config:
            patterns = _compile_motifs(motif_config)
            motifs   = find_motifs(pdb_text, motif_config)
            for mname, resi_list in motifs.items():
                mcol     = patterns[mname][2]
                resi_str = ",".join(str(r) for r in resi_list)
                view.addStyle({"model": -1, "resi": resi_str},
                              {"stick": {"color": mcol, "radius": 0.25}})

    view.zoomTo()
    return view


def rmsd_table(
    pdb_paths: dict[str, str | Path],
    ref_label: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Compute pairwise C-alpha RMSD between structures using Biopython.
    If ref_label given, only returns rows vs that reference.
    """
    import numpy as np
    import pandas as pd
    from Bio.PDB import PDBParser

    parser   = PDBParser(QUIET=True)
    ca_atoms = {}
    for label, path in pdb_paths.items():
        struct = parser.get_structure(label, str(path))
        ca_atoms[label] = np.array([
            a.get_vector().get_array()
            for a in struct.get_atoms()
            if a.get_name() == "CA"
        ])

    rows   = []
    labels = list(pdb_paths.keys())
    pairs  = ([(k, ref_label) for k in labels if k != ref_label]
              if ref_label else
              [(labels[i], labels[j]) for i in range(len(labels))
               for j in range(i + 1, len(labels))])

    for a, b in pairs:
        ca_a, ca_b = ca_atoms[a], ca_atoms[b]
        n = min(len(ca_a), len(ca_b))
        rmsd = float(np.sqrt(((ca_a[:n] - ca_b[:n]) ** 2).sum(axis=1).mean()))
        rows.append({"structure_A": a, "structure_B": b,
                     "RMSD_Angstrom": round(rmsd, 3), "n_residues": n})

    return pd.DataFrame(rows)
