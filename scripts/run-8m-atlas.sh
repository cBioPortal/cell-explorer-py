#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cell2zarr-8m
#SBATCH --output=slurm-%j.out

cd ~/cell2zarr
uv run cell2zarr /path/to/input/atlas.h5ad /path/to/output/atlas.zarr --two-phase --encoding-config configs/encoding-config-8m-atlas.json
