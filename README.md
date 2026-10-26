# HHL

Prof. Li's unstable-shear Navier-Stokes test, assembled as linear systems for
an HHL (quantum linear solver) platform. Each MATLAB script has a Python port
that produces the same results.

```
matlab/   original MATLAB scripts
python/   Python ports (run with uv)
output/   generated .mat files (git-ignored; created on first run)
```

| Script | What it builds |
|---|---|
| `prof_li_hhl_minimal` | One frozen-velocity slab as the full system `A z = b` (unknowns `[Ustar; p; U]` per step), plus the Hermitian embedding `H x = rhs`. Writes `output/prof_li_hhl_case.mat`. |
| `prof_li_hhl_reduced_multislab` | Projects velocity onto a divergence-free basis `Z` (`U = Z a`), applies the fifth-order transform `A5 = I - X^5`, and solves one frozen slab at a time, rebuilding `B(V)`, `K` and `A5` from the last velocity before each new slab. Writes `output/hhl_reduced_cases/slab_###.mat` (HHL inputs `H`, `rhsH` plus `a_classical`) and `summary.mat`. |

## Python

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv run python/prof_li_hhl_minimal.py
uv run python/prof_li_hhl_reduced_multislab.py
```

uv creates `.venv` and installs numpy/scipy from `uv.lock` on first run.

## MATLAB

Run `prof_li_hhl_minimal` or `prof_li_hhl_reduced_multislab` from the
`matlab/` folder. Output goes to the same `output/` folder as Python.

## Plugging in HHL

In `prof_li_hhl_reduced_multislab`, set `useHHLadapter = true` (MATLAB) /
`useHHLadapter = True` (Python) and implement `student_hhl_adapter`. It must
return the correctly scaled `a`, not just a normalized quantum state.
