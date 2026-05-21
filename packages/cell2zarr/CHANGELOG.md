# CHANGELOG

<!-- version list -->

## v1.0.0 (2026-05-21)

### Bug Fixes

- **strata**: Re-consolidate .zmetadata after writes so production readers see new groups
  ([`8751765`](https://github.com/cBioPortal/cell-explorer-py/commit/87517658cffd04c69452964a0c331b5bfb2b6f58))

- **strata**: Size stratum_keys dtype to longest label, not hardcoded <U32
  ([`1c39459`](https://github.com/cBioPortal/cell-explorer-py/commit/1c394590ea5bf904374a3275fa931fcbc403152e))

### Build System

- Add scipy and pyyaml as cell2zarr runtime deps for strata-build
  ([`4cfbe2b`](https://github.com/cBioPortal/cell-explorer-py/commit/4cfbe2b482708cb022738b9b580094ef48178a83))

### Chores

- Switch pytest to importlib import mode for monorepo
  ([`b81b6b2`](https://github.com/cBioPortal/cell-explorer-py/commit/b81b6b29d0a26b7d93df2d26a212e8b0012439c1))

### Documentation

- Document build-strata subcommand in cell2zarr README
  ([`5d7d7fc`](https://github.com/cBioPortal/cell-explorer-py/commit/5d7d7fc1d93ab4d8c01757cc332a527394a0059c))

- **strata**: Correct consolidated-metadata wording (zarr v3 inlines in zarr.json, no .zmetadata)
  ([`a0a7c2f`](https://github.com/cBioPortal/cell-explorer-py/commit/a0a7c2fd5ce267b433b3a7cf13a0951544c0199b))

### Features

- Add AtomicTable dataclass and compute_strata_mapping
  ([`1d50ab9`](https://github.com/cBioPortal/cell-explorer-py/commit/1d50ab9839c73f04a691f641638326670f939f57))

- Add build_strata orchestrator tying validate + atomic + coarse + io
  ([`f063b70`](https://github.com/cBioPortal/cell-explorer-py/commit/f063b70898621a59dc6de87d4b3e9ffa0f6b4ec5))

- Add derive_coarse via sparse-matmul axis collapse
  ([`3bb931f`](https://github.com/cBioPortal/cell-explorer-py/commit/3bb931f210fc82ab7e9501605b0eb0a0d51d12e9))

- Add strata.io read primitives (open_dataset, read_obs, read_x_block, has_strata)
  ([`2774a28`](https://github.com/cBioPortal/cell-explorer-py/commit/2774a282774a85bc81eefcda5b7961749abed3de))

- Add strata.io write primitives (write_atomic, write_coarse) with completion-marker pattern
  ([`267ef1a`](https://github.com/cBioPortal/cell-explorer-py/commit/267ef1af52c7eb636048c5704d4012939c3409c9))

- Add strata.validate preflight checks (obs columns, cardinality, null-rate)
  ([`a429a23`](https://github.com/cBioPortal/cell-explorer-py/commit/a429a232b479a40b21e418400ee97b38f0caaea4))

- Add StrataConfig pydantic model with YAML loader
  ([`c76b0c6`](https://github.com/cBioPortal/cell-explorer-py/commit/c76b0c60886768e9512439dfab98da25906aec1b))

- Add StrataError exception hierarchy
  ([`85a5552`](https://github.com/cBioPortal/cell-explorer-py/commit/85a55527150b9f075632bfeeeeca72a5d83c16c6))

- Implement build_atomic via sparse-matmul accumulation
  ([`76fd8e5`](https://github.com/cBioPortal/cell-explorer-py/commit/76fd8e5d722022137c3867242305a3c22e4e9456))

- Register cell2zarr build-strata click subcommand
  ([`698b5d6`](https://github.com/cBioPortal/cell-explorer-py/commit/698b5d6c3bbc351bdee37f036d3b365f0d331201))

- Scaffold cell2zarr.strata submodule
  ([`c6ffd3f`](https://github.com/cBioPortal/cell-explorer-py/commit/c6ffd3f78a94c8d213baad5b3f5859ca088ae51d))

- **cell2zarr**: Explicit chunks + shards + zstd for atomic strata arrays
  ([`c01cecd`](https://github.com/cBioPortal/cell-explorer-py/commit/c01cecd21811647be5b841cd1fc852b99c086cc8))

- **strata**: Add --add-coarse-only mode to derive new coarse tables without re-reading X
  ([`15d3eb2`](https://github.com/cBioPortal/cell-explorer-py/commit/15d3eb2f403ff281c5f8844f97c40a9f665a5395))

### Testing

- Add PBMC3K integration tests for build-strata
  ([`5c5a0a2`](https://github.com/cBioPortal/cell-explorer-py/commit/5c5a0a20bdffe7eee63e80792baeba1790194c25))

- Add tiny_anndata_zarr and pbmc3k_zarr fixtures for strata tests
  ([`2ff033c`](https://github.com/cBioPortal/cell-explorer-py/commit/2ff033c39b06953d8df03488ad955ff782846259))

- **cell2zarr**: Assert atomic chunks + shards + coarse-isolation (failing)
  ([`097654d`](https://github.com/cBioPortal/cell-explorer-py/commit/097654dc086ef17c1fc1c6dc859f47f74729e8c5))

- **cell2zarr**: Build strata fixtures as v3 zarr matching production layout
  ([`89d58bc`](https://github.com/cBioPortal/cell-explorer-py/commit/89d58bcf8fa0559bd51541f44eaa850d30a9517d))

- **cell2zarr**: Tighten atomic-sharding contract tests per code review
  ([`1c2c600`](https://github.com/cBioPortal/cell-explorer-py/commit/1c2c600c42065138de26832774b64a6104b520b6))


## v0.1.0 (2026-04-02)

- Initial Release
