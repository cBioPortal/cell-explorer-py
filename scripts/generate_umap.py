"""Generate UMAP embedding for an h5ad file.

Pipeline: normalize → log1p → HVG → PCA → neighbors → UMAP
Writes a new h5ad with X_umap added to obsm.

Usage:
    python scripts/generate_umap.py input.h5ad output.h5ad
    python scripts/generate_umap.py input.h5ad output.h5ad --n-top-genes 3000 --n-neighbors 30
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import scanpy as sc


def generate_umap(
    input_path: Path,
    output_path: Path,
    n_top_genes: int = 3000,
    n_comps: int = 50,
    n_neighbors: int = 30,
    n_pcs: int = 50,
):
    start = time.time()

    print(f"Reading {input_path}...", flush=True)
    adata = sc.read_h5ad(input_path)
    print(f"Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes", flush=True)

    print("Normalizing...", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"Finding top {n_top_genes} highly variable genes...", flush=True)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    print(f"HVG shape: {adata_hvg.shape}", flush=True)

    # Free full adata before expensive steps
    del adata
    gc.collect()
    print("Freed full adata from memory", flush=True)

    print(f"Computing PCA (n_comps={n_comps})...", flush=True)
    sc.pp.pca(adata_hvg, n_comps=n_comps, zero_center=False)

    print(f"Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs})...", flush=True)
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs)

    print("Computing UMAP...", flush=True)
    sc.tl.umap(adata_hvg)

    # Save computed results before freeing adata_hvg
    pca_coords = adata_hvg.obsm["X_pca"]
    umap_coords = adata_hvg.obsm["X_umap"]
    neighbors_connectivities = adata_hvg.obsp["connectivities"].copy()
    neighbors_distances = adata_hvg.obsp["distances"].copy()
    neighbors_params = dict(adata_hvg.uns["neighbors"])
    pca_variance = adata_hvg.uns["pca"].copy() if "pca" in adata_hvg.uns else None
    hvg_var = adata_hvg.var.copy()
    del adata_hvg
    gc.collect()

    # Reload original to attach all results
    print(f"Reloading {input_path} to attach results...", flush=True)
    adata = sc.read_h5ad(input_path)

    # Embeddings
    adata.obsm["X_pca"] = pca_coords
    adata.obsm["X_umap"] = umap_coords

    # Neighbors graph (enables re-running UMAP without recomputing neighbors)
    adata.obsp["connectivities"] = neighbors_connectivities
    adata.obsp["distances"] = neighbors_distances
    adata.uns["neighbors"] = neighbors_params

    # PCA variance info
    if pca_variance is not None:
        adata.uns["pca"] = pca_variance

    # HVG annotations (merge into var)
    for col in ["highly_variable", "means", "dispersions", "dispersions_norm"]:
        if col in hvg_var.columns:
            adata.var[col] = False if col == "highly_variable" else 0.0
            adata.var.loc[hvg_var.index, col] = hvg_var[col]

    del pca_coords, umap_coords, neighbors_connectivities, neighbors_distances
    gc.collect()

    import anndata as ad
    ad.settings.allow_write_nullable_strings = True
    print(f"Writing {output_path}...", flush=True)
    adata.write_h5ad(output_path)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP embedding for an h5ad file")
    parser.add_argument("input", type=Path, help="Input h5ad file")
    parser.add_argument("output", type=Path, help="Output h5ad file (new file with UMAP added)")
    parser.add_argument("--n-top-genes", type=int, default=3000, help="Number of highly variable genes. Default: 3000")
    parser.add_argument("--n-comps", type=int, default=50, help="Number of PCA components. Default: 50")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors for kNN graph. Default: 30")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs for neighbors. Default: 50")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.output.exists():
        print(f"Error: {args.output} already exists", file=sys.stderr)
        sys.exit(1)

    generate_umap(
        input_path=args.input,
        output_path=args.output,
        n_top_genes=args.n_top_genes,
        n_comps=args.n_comps,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
    )


if __name__ == "__main__":
    main()
