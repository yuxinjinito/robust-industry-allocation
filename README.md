# Robust asset allocation when expected returns are unreliable

Long-only allocation across the 30 U.S. industry portfolios of the Kenneth French Data Library,
with a Bertsimas-Sim robust linear program that lets an adversary lower the estimated mean
return of up to Γ industries. The project started as the final project of CS 524 (Introduction
to Optimization, UW-Madison, summer 2026), a three-person team project that I led: I built the
model, derived the LP dual, wrote the Julia/JuMP code and ran the experiments. After the course
I extended it in two rounds, described in `writeup/addendum.md`.

## What is here

| path | contents |
|---|---|
| `baseline/Final_Report.ipynb` | the course notebook as submitted: model, dual derivation, brute-force check, holdout test, 31 rolling windows, sensitivity |
| `baseline/nb.jl`, `baseline/saved.txt` | the notebook's code cells and printed outputs, used to check that the extension code reproduces the course results exactly |
| `data/30_Industry_Portfolios_Daily.csv` | Ken French "30 Industry Portfolios (Daily)", May 2026 snapshot, md5 `06f3af680bb76a42ac2522c76b9203f1` |
| `src/harness.jl` | shared code: data loading, the box LP, the ellipsoidal SOCP, the selection rule, scoring with turnover and costs |
| `src/run.jl`, `src/tables.jl` | round 1: seven allocation rules on the same 31 rolling windows |
| `src/injection.jl`, `src/injection_tables.jl` | round 2: known signals injected into real market noise to measure a detection threshold |
| `results/` | the outputs of both rounds: `summary.json` files, tables, figures, run log |
| `writeup/addendum.md` | what the two rounds found, with every prediction marked held or refuted |

## Findings

- Against a portfolio that trusts its estimated means completely (all capital in the single
  industry with the highest estimate), the robust portfolio cut the median maximum drawdown
  from 37.5% to 26.8% across 31 three-year test windows from 1934 to 2026, with no loss of
  return (`results/tables.md`, T1 and T5).
- Against equal weight it is comparable, not better. Using the covariance (an ellipsoidal
  uncertainty set), shrinking means toward a CAPM-implied target, widening the set for means
  that drift, or charging transaction costs did not change that (T1).
- With weights that sum to one, shrinking every mean toward a common value is the same as
  widening the uncertainty set, so it cannot act as a separate repair step (addendum, section 1).
- When known differences between industry premia are injected into the real returns, the robust
  rules beat equal weight in most simulated histories once those premia differ by about 2% a
  year (`results/injection_tables.md`).

## Reproducing

Julia 1.12.6. The environment is pinned in `Manifest.toml`.

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. src/run.jl && julia --project=. src/tables.jl
julia --project=. src/injection.jl && julia --project=. src/injection_tables.jl
```

Round 1 takes about a minute and round 2 about twenty. Both scripts first check that the shared
code reproduces every printed line of the course notebook, and stop if it does not. They also
refuse to run with uncommitted changes in `src/` or the environment, so every `summary.json`
names the commit that produced it.

The data are from the Kenneth R. French Data Library
(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). The library
revises recent history from month to month, so the snapshot used here is included.
