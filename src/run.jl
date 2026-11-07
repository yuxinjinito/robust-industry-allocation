# The one command: julia --project=. src/run.jl
# Runs the gates in order, stops at the first failure, then scores every arm on the
# holdout and on every rolling window, and writes results/summary.json. Statistics over
# windows (counts, sign tests, predictions) are computed from summary.json by tables.jl.

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "harness.jl"))
include(joinpath(ROOT, "src", "rgate.jl"))
using JSON, Dates
import Pkg
const GATES_ONLY = "--gates-only" in ARGS

git(args...) = readchomp(Cmd(`git -C $ROOT $(collect(args))`))
gate_fail(k, msg) = (println("GATE $k FAILED: $msg"); exit(1))
gates = Dict{String,Any}()

# ---- gate 1: code and environment committed and clean, so the summary names the code that made it
dirty = git("status", "--porcelain", "--", "src", "Project.toml", "Manifest.toml")
isempty(dirty) || gate_fail(1, "uncommitted changes in src or the environment:\n$dirty")
src_commit = git("rev-parse", "HEAD")
gates["1_clean"] = "pass"
println("gate 1 pass: code $(src_commit[1:7])")

dates, industries, R = load_vw_daily(joinpath(ROOT, "data", "30_Industry_Portfolios_Daily.csv"))

# ---- gate 2: R-gate
ok, nlines, bad = rgate(dates, industries, R, joinpath(ROOT, "baseline", "saved.txt"))
ok || gate_fail(2, "R-gate mismatch\n" * join(bad, "\n"))
gates["2_rgate"] = "pass, $nlines baseline lines identical"
println("gate 2 pass: $nlines baseline lines identical")

# ---- gate 3: data gate on mu_hat of every fit block. The mu_w part runs inside run_arms,
# where it is known whether RBOX or RELL sit at Gamma* = 0 or kappa* = 0.
hm = holdout_masks(dates)
wins = rolling_windows(dates)
length(wins) == 31 || gate_fail(3, "expected 31 rolling windows, got $(length(wins))")
min_gap = Inf
for (lab, w) in vcat([("holdout", hm)], [("rolling $(w.y0)", w) for w in wins])
    for (part, mask) in (("train", w.tr), ("refit", w.tr .| w.va))
        g = top_gap(vec(mean(R[mask, :], dims=1)))
        g > GAP_MIN || gate_fail(3, "top-two mu_hat gap $g at $lab $part")
        global min_gap = min(min_gap, g)
    end
end
gates["3_data"] = "pass, smallest top-two mu_hat gap $(min_gap) per day"
println("gate 3 pass: smallest top-two gap $(min_gap)")

# ---- gate 4: E1, SOCP against the mean-variance QP on the holdout refit inputs
ftr_h, fref_h = window_fits(dates, R, hm)
e1 = Dict{String,Any}()
for κ in (0.5, 2.0, 5.0)
    x = clean(solve_ell(fref_h, fref_h.mu, κ; ctx = "E1", strict = true))
    λ = κ / (2 * sqrt(x' * (SCALE^2 .* fref_h.Omega) * x))
    xq = clean(solve_qp(fref_h, fref_h.mu, λ))
    dmax = maximum(abs.(x .- xq))
    e1["kappa_$κ"] = Dict("lambda" => λ, "max_weight_diff" => dmax)
    @printf("E1 kappa = %.1f: lambda = %.6g, max weight difference %.3g\n", κ, λ, dmax)
    dmax < 1e-5 || gate_fail(4, "E1 difference $dmax at kappa $κ")
end
gates["4_E1"] = e1
println("gate 4 pass")

# ---- gate 5: leakage audit for every arm on the holdout
books_h = run_arms(ftr_h, fref_h, R[hm.va, :]; ctx = "holdout")
keep = dates .<= 20221231
dk, Rk = dates[keep], R[keep, :]
hk = holdout_masks(dk)
ftr_k, fref_k = window_fits(dk, Rk, hk)
books_k = run_arms(ftr_k, fref_k, Rk[hk.va, :]; ctx = "holdout truncated")
leak = Dict{String,Any}()
for a in ARMS
    same_h = books_h[a].h == books_k[a].h
    dmax = maximum(abs.(books_h[a].x .- books_k[a].x))
    leak[string(a)] = Dict("same_hyperparameter" => same_h, "max_weight_diff" => dmax)
    (same_h && dmax <= 1e-12) || gate_fail(5, "leakage audit fails for $a: h $(books_h[a].h) vs $(books_k[a].h), diff $dmax")
end
gates["5_leakage"] = leak
println("gate 5 pass: all seven arms identical on truncated data")
GATES_ONLY && (println("--gates-only: stopping before any book is scored"); exit(0))

# ---- arms on every window
book_record(b) = Dict{String,Any}("h" => b.h, "x" => b.x, "fixed" => get(b, :fixed, nothing))

function window_record(lab, ftr, fref, books, Rte, dte; shuffled = nothing)
    rec = Dict{String,Any}("label" => lab, "test_days" => size(Rte, 1),
        "train" => Dict("drift_ratio" => ftr.drift_ratio, "drift_ratio_shuffled" => shuffled,
                        "drift_excess" => shuffled === nothing ? nothing : ftr.drift_ratio / shuffled,
                        "wJ" => ftr.wJ, "wJ_inflation" => 1 / (1 - ftr.wJ),
                        "mbar" => ftr.mbar, "mbar_sign" => sign(ftr.mbar)),
        "ell_kappa0_vs_nom_max_diff" => maximum(abs.(clean(build_ell(fref, 0.0; ctx = lab)) .- books[:NOM].x)),
        "arms" => Dict{String,Any}())
    for a in ARMS
        r = book_record(books[a])
        r["score"] = score(Rte, dte, books[a].x)
        r["same_as_EW"] = maximum(abs.(books[a].x .- books[:EW].x)) < 1e-9
        rec["arms"][string(a)] = r
    end
    return rec
end

holdout_rec = window_record("holdout 2023-2025", ftr_h, fref_h, books_h, R[hm.te, :], dates[hm.te])
println("holdout scored")

Random.seed!(524)
rng = Random.default_rng()
window_recs = []
for w in wins
    ftr, fref = window_fits(dates, R, w)
    shuf = shuffled_drift_ratio(R[w.tr, :], dates[w.tr], rng)
    books = run_arms(ftr, fref, R[w.va, :]; ctx = "rolling $(w.y0)")
    push!(window_recs, window_record("test $(w.y0)-$(w.y0 + 2)", ftr, fref, books,
                                     R[w.te, :], dates[w.te]; shuffled = shuf))
    @printf("window %d: Γ*=%s κ*=%.3g w*(RBOX)=%.3g w*(RELL)=%.3g Γ*(DRIFT)=%s\n", w.y0,
            books[:BOX].h, books[:ELL].h, books[:RBOX].h, books[:RELL].h, books[:DRIFT].h)
end

# ---- gate 6: sanity on every book
for rec in vcat([holdout_rec], window_recs), a in ARMS
    x = rec["arms"][string(a)]["x"]
    s = rec["arms"][string(a)]["score"]
    abs(sum(x) - 1) < 1e-9 || gate_fail(6, "weights of $a at $(rec["label"]) sum to $(sum(x))")
    any(isnan, x) && gate_fail(6, "NaN weight in $a at $(rec["label"])")
    any(v -> v isa Float64 && isnan(v), values(s)) && gate_fail(6, "NaN score in $a at $(rec["label"])")
end
length(window_recs) == 31 || gate_fail(6, "windows missing")
gates["6_sanity"] = "pass"
println("gate 6 pass; Clarabel ALMOST_OPTIMAL solves: $(length(ALMOST_LOG))")

pkgver = Dict(p.name => string(p.version) for p in values(Pkg.dependencies())
              if p.version !== nothing && p.name in ("JuMP", "HiGHS", "HiGHS_jll", "Clarabel",
                                                     "MathOptInterface", "Statistics", "JSON"))
summary = Dict{String,Any}(
    "header" => Dict("code_commit" => src_commit,
                     "run_date" => string(Dates.now()), "julia" => string(VERSION),
                     "packages" => pkgver,
                     "seed" => 524, "nperm" => 200),
    "industries" => industries,
    "grids" => Dict("Gamma" => GRID_G, "kappa" => GRID_K, "w" => GRID_W, "costs_bps" => COSTS_BPS),
    "gates" => gates,
    "almost_optimal" => ALMOST_LOG,
    "holdout" => holdout_rec,
    "windows" => window_recs)
mkpath(joinpath(ROOT, "results"))
open(joinpath(ROOT, "results", "summary.json"), "w") do io
    write(io, JSON.json(summary; pretty = true))
end
println("wrote results/summary.json")
