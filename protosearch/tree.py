"""MAFFT alignment + FastTree / IQ-TREE2 phylogenetics."""

from __future__ import annotations
import subprocess
from pathlib import Path


def align(
    fasta_path:  str | Path,
    output_path: str | Path,
    flags:       str = "--auto --thread 8 --quiet",
    mafft_bin:   str = "mafft",
) -> Path:
    """Run MAFFT. Returns path to aligned FASTA."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [mafft_bin] + flags.split() + [str(fasta_path)],
        capture_output=True, text=True, check=True,
    )
    output_path.write_text(result.stdout)
    return output_path


def fasttree(
    aligned_path: str | Path,
    output_path:  str | Path,
    model:        str = "lg",
    fasttree_bin: str = "FastTree",
) -> Path:
    """Run FastTree. Returns path to Newick tree."""
    output_path = Path(output_path)
    model_flag  = f"-{model}"
    result = subprocess.run(
        [fasttree_bin, model_flag, "-quiet", str(aligned_path)],
        capture_output=True, text=True,
    )
    output_path.write_text(result.stdout)
    return output_path


def iqtree(
    aligned_path:  str | Path,
    output_dir:    str | Path,
    prefix:        str,
    model:         str  = "LG+G4",
    bootstrap:     int  = 1000,
    threads:       int  = 8,
    asr:           bool = True,
    iqtree_bin:    str  = "iqtree2",
) -> dict[str, Path]:
    """
    Run IQ-TREE2 with optional ancestral state reconstruction.
    Returns dict of output file paths: treefile, state, iqtree, log.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_prefix = output_dir / prefix

    cmd = [
        iqtree_bin,
        "-s", str(aligned_path),
        "-m", model,
        "-B", str(bootstrap),
        "-T", str(threads),
        "--redo",
        "--prefix", str(full_prefix),
    ]
    if asr:
        cmd.append("--ancestral")

    subprocess.run(cmd, check=True)

    return {
        "treefile": full_prefix.with_suffix(".treefile"),
        "state":    Path(str(full_prefix) + ".state"),
        "iqtree":   Path(str(full_prefix) + ".iqtree"),
        "log":      Path(str(full_prefix) + ".log"),
    }


def align_and_tree(
    fasta_path:   str | Path,
    output_prefix: str | Path,
    mafft_flags:  str = "--auto --thread 4 --quiet",
    model:        str = "lg",
) -> tuple[Path, Path]:
    """MAFFT align + FastTree. Returns (aligned_faa, tree_newick)."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    aligned  = Path(str(output_prefix) + "_aligned.faa")
    tree_out = Path(str(output_prefix) + ".tree")
    align(fasta_path, aligned, flags=mafft_flags)
    fasttree(aligned, tree_out, model=model)
    return aligned, tree_out


def run_iqtree_asr(
    aligned_path: str | Path,
    output_dir:   str | Path,
    model:        str = "LG+G4",
    bootstrap:    int = 1000,
    threads:      int = 4,
) -> dict[str, Path]:
    """IQ-TREE2 with ancestral state reconstruction. Returns dict of output paths."""
    prefix = Path(aligned_path).stem
    return iqtree(aligned_path, output_dir, prefix,
                  model=model, bootstrap=bootstrap, threads=threads, asr=True)


def build_cluster_trees(
    cluster_fastas: dict[str, Path],   # {cluster_name: fasta_path}
    output_root:    str | Path,
    mafft_flags:    str = "--auto --thread 8 --quiet",
    fasttree_model: str = "lg",
    skip_existing:  bool = True,
) -> dict[str, tuple[Path, Path]]:
    """
    Align + tree for multiple clusters in one call.
    Returns {cluster_name: (aligned_faa, tree_newick)}.
    """
    output_root = Path(output_root)
    results = {}
    for name, fasta in cluster_fastas.items():
        out_dir   = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        aln_path  = out_dir / f"{name}_aligned.faa"
        tree_path = out_dir / f"{name}.tree"

        if skip_existing and aln_path.exists() and tree_path.exists():
            print(f"  {name}: already built, skipping")
            results[name] = (aln_path, tree_path)
            continue

        print(f"  {name}: aligning ...")
        align(fasta, aln_path, flags=mafft_flags)
        print(f"  {name}: building tree ...")
        fasttree(aln_path, tree_path, model=fasttree_model)
        results[name] = (aln_path, tree_path)
        print(f"  {name}: done")

    return results
