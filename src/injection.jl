# Signal injection into real noise. One command:
#   julia --project=. src/injection.jl            full run
#   julia --project=. src/injection.jl --gates-only   gates 1-7 and the data-gate count, no arm scored
# Reuses src/harness.jl unchanged. Writes results/injection_summary.json; statistics over
# histories (p_beat, isotonic fits, s*, bootstraps, predictions) are computed from that file by
# src/injection_tables.jl.

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "harness.jl"))
include(joinpath(ROOT, "src", "rgate.jl"))
using JSON, Dates
import Pkg
const GATES_ONLY = "--gates-only" in ARGS
const T0 = time()
say(x...) = (println(round(time() - T0; digits = 1), "s  ", x...); flush(stdout))

git(args...) = readchomp(Cmd(`git -C $ROOT $(collect(args))`))
gate_fail(k, msg) = (println("GATE $k FAILED: $msg"); exit(1))
gates = Dict{String,Any}()

# ---- gate 1: code and environment committed and clean
dirty = git("status", "--porcelain", "--", "src", "Project.toml", "Manifest.toml")
isempty(dirty) || gate_fail(1, "uncommitted changes in src or the environment:\n$dirty")
code_commit = git("rev-parse", "HEAD")
pkgver = Dict(p.name => string(p.version) for p in values(Pkg.dependencies())
              if p.version !== nothing && p.name in ("JuMP", "HiGHS", "HiGHS_jll", "Clarabel",
                                                     "MathOptInterface", "Statistics", "JSON"))
haskey(pkgver, "Clarabel") || gate_fail(1, "Clarabel not in the environment")
gates["1_clean"] = "pass"
say("gate 1 pass: code $(code_commit[1:7]), Clarabel $(pkgver["Clarabel"])")

dates, industries, R = load_vw_daily(joinpath(ROOT, "data", "30_Industry_Portfolios_Daily.csv"))

# ---- gate 2
ok, nlines, bad = rgate(dates, industries, R, joinpath(ROOT, "baseline", "saved.txt"))
ok || gate_fail(2, "R-gate mismatch\n" * join(bad, "\n"))
gates["2_rgate"] = "pass, $nlines baseline lines identical"
say("gate 2 pass")

# ---- the years Y and their rows
const YEARS = collect(1927:2025)
ymask = (dates .÷ 10000 .>= 1927) .& (dates .÷ 10000 .<= 2025)
Ry, dy = R[ymask, :], dates[ymask]
yr = dy .÷ 10000
@assert length(unique(yr)) == 99
rows_of = Dict(y => findall(==(y), yr) for y in YEARS)
colsum_y = Dict(y => vec(sum(Ry[rows_of[y], :], dims = 1)) for y in YEARS)
nrow_y = Dict(y => length(rows_of[y]) for y in YEARS)
const S_GRID = [0.0, 0.5, 1, 2, 3, 4, 6, 8, 10] ./ 100
const NS = length(S_GRID)

# ---- patterns
z_dense(r) = (r .- 15.5) ./ 8.655
const C_SPARSE = 1 / sqrt((1 / 30) * (29 / 30))
z_sparse(k) = C_SPARSE .* ([i == k ? 1.0 : 0.0 for i in 1:30] .- 1 / 30)

# real-data relative yearly means, used by realdir, s_hat and gate 7
function yearly_rel_means(Rw, yrw)
    u = Rw .- mean(Rw, dims = 2)
    return reduce(vcat, [252 .* mean(u[yrw .== y, :], dims = 1) for y in YEARS])   # 99 x 30
end
ubar = yearly_rel_means(Ry, yr)
mt = vec(mean(ubar, dims = 1))
z_real = mt ./ std(mt; corrected = false)

# ---- histories
struct History
    years::Vector{Int}       # 11 years in history order: train 1-6, validation 7-8, test 9-11
    r::Vector{Int}           # dense permutation
    k::Int                   # sparse industry
    label::String
end
Random.seed!(2027)
primary = [begin
    yrs = YEARS[randperm(99)[1:11]]
    r = randperm(30); k = rand(1:30)
    History(yrs, r, k, "primary $d")
end for d in 1:200]
Random.seed!(2028)
contig = [begin
    r = randperm(30); k = rand(1:30)
    History(collect(y0:y0+10), r, k, "contiguous $y0")
end for y0 in 1927:2015]

train_years(h) = h.years[1:6]; val_years(h) = h.years[7:8]
fit_years(h) = h.years[1:8];   test_years(h) = h.years[9:11]
rows(ys) = reduce(vcat, [rows_of[y] for y in ys])

# demeaning constant over Y \ E, row-weighted
function demean_const(h)
    E = test_years(h)
    tot = sum(colsum_y[y] for y in YEARS if !(y in E))
    n = sum(nrow_y[y] for y in YEARS if !(y in E))
    c = tot ./ n
    return c, mean(c)
end

# R(s) on a row set of history h for pattern z
function world_rows(h, rr, s, z)
    c, cb = demean_const(h)
    return Ry[rr, :] .- (c .- cb)' .+ (s / 252) .* z'
end

patterns_of(scheme) = scheme == "primary" ? ["dense", "sparse", "realdir"] : ["dense", "sparse"]
zof(h, p) = p == "dense" ? z_dense(h.r) : p == "sparse" ? z_sparse(h.k) : z_real

# ---- gate 3: null check over Y \ E, every history
let worst = 0.0
    for h in vcat(primary, contig)
        c, cb = demean_const(h)
        rr = rows([y for y in YEARS if !(y in test_years(h))])
        m0 = vec(mean(Ry[rr, :] .- (c .- cb)', dims = 1))
        worst = max(worst, maximum(abs.(m0 .- cb)))
    end
    worst <= 1e-12 || gate_fail(3, "null mean off by $worst")
    gates["3_null"] = "pass, worst $worst"
    say("gate 3 pass: worst $worst")
end

# ---- gate 4: injection check on the first primary history at s = 5%
let h = primary[1], worst = 0.0
    rr = rows([y for y in YEARS if !(y in test_years(h))])
    c, cb = demean_const(h)
    for z in (z_dense(h.r), z_sparse(h.k))
        m = vec(mean(world_rows(h, rr, 0.05, z), dims = 1))
        worst = max(worst, maximum(abs.((m .- cb) .- 0.05 .* z ./ 252)))
    end
    worst <= 1e-12 || gate_fail(4, "injection off by $worst")
    gates["4_injection"] = "pass, worst $worst"
    say("gate 4 pass: worst $worst")
end

say("starting gate 5")
# ---- gate 5 and headroom
SigY = cov(Ry)
mbarY = mean(vec(mean(Ry, dims = 1)))
# HiGHS's QP solver loops without converging on the min-variance problem at the raw daily
# scale (Hessian entries near 1e-4; 4 million iterations in 15 s, probe of 2026-11-06), and
# converges in 14 iterations with the covariance times 1e4. A positive rescale leaves the
# argmin unchanged. Both QPs carry a time limit so a stall stops the run through check_status.
const QP_SCALE = 1e4
function headroom_book(mu)
    m = Model(HiGHS.Optimizer); set_silent(m); set_time_limit_sec(m, 30.0)
    @variable(m, y[1:30] >= 0)
    @constraint(m, mu' * y == 1)
    @objective(m, Min, y' * (QP_SCALE .* SigY) * y)
    optimize!(m); check_status(m, :highs, "headroom")
    yv = max.(value.(y), 0.0)
    return yv ./ sum(yv)
end
function minvar_book()
    m = Model(HiGHS.Optimizer); set_silent(m); set_time_limit_sec(m, 30.0)
    @variable(m, x[1:30] >= 0)
    @constraint(m, sum(x) == 1)
    @objective(m, Min, x' * (QP_SCALE .* SigY) * x)
    optimize!(m); check_status(m, :highs, "minvar")
    return clean(value.(x))
end
popsharpe(mu, x) = sqrt(252) * dot(mu, x) / sqrt(x' * SigY * x)
ew = fill(1 / 30, 30)
let x0 = headroom_book(fill(mbarY, 30)), xm = minvar_book()
    v0, vm = x0' * SigY * x0, xm' * SigY * xm
    rel = abs(v0 - vm) / vm
    rel <= 1e-6 || gate_fail(5, "headroom at s = 0 variance differs from min-variance by $rel relative")
    gates["5_headroom_minvar_rel"] = rel
end
headroom = Dict{String,Any}()
for p in ["dense", "sparse", "realdir"]
    hs, es = Float64[], Float64[]
    for (j, s) in enumerate(S_GRID)
        zs = p == "realdir" ? [z_real] : [zof(h, p) for h in primary]
        vals = Float64[]; evals = Float64[]
        for z in zs
            mu = mbarY .+ s .* z ./ 252
            x = headroom_book(mu)
            ps, pe = popsharpe(mu, x), popsharpe(mu, ew)
            ps >= pe - 1e-9 || gate_fail(5, "headroom below EW at $p s=$s")
            push!(vals, ps); push!(evals, pe)
        end
        push!(hs, mean(vals)); push!(es, mean(evals))
    end
    headroom[p] = Dict("oracle_pop_sharpe" => hs, "ew_pop_sharpe" => es)
    say("headroom done for ", p)
end
gates["5_headroom"] = "pass"
say("gate 5 pass")

# ---- gate 6: disjointness
for h in primary
    @assert isempty(intersect(train_years(h), val_years(h))) && isempty(intersect(fit_years(h), test_years(h)))
end
gates["6_disjoint"] = "pass"

# ---- s_hat machinery, on a 99 x 30 matrix of annualized yearly relative means
vobs(ub) = mean(vec(mean(ub, dims = 1)) .^ 2)
noise_iid(ub) = mean(vec(var(ub, dims = 1)) ./ size(ub, 1))
est(ub) = vobs(ub) - noise_iid(ub)
pct(v, q) = sort(v)[clamp(ceil(Int, q * length(v)), 1, length(v))]   # inverse-cdf percentile
function interval_iid(ub, rng; nb = 1000)
    N = noise_iid(ub)
    sb = [sqrt(max(0.0, vobs(ub[rand(rng, 1:99, 99), :]) - 2N)) for _ in 1:nb]
    return pct(sb, 0.05), pct(sb, 0.95)
end

# ---- gate 7: estimator checks in a world with known spread
let
    cy = vec(mean(reduce(vcat, [mean(Ry[rows_of[y], :], dims = 1) for y in YEARS]), dims = 1))
    Wd = Ry .- (cy .- mean(cy))'
    Random.seed!(2029)
    zg = z_dense(randperm(30))
    rng = Random.default_rng()
    g7 = Dict{String,Any}()
    first_s2 = nothing; outer_s2 = Vector{Vector{Int}}()
    for s in (0.0, 0.02, 0.04)
        ub = yearly_rel_means(Wd .+ (s / 252) .* zg', yr)
        vals = Float64[]
        for b in 1:500
            idx = rand(rng, 1:99, 99)
            push!(vals, est(ub[idx, :]))
            s == 0.02 && push!(outer_s2, idx)
        end
        mcse = std(vals) / sqrt(500)
        g7["line1_s$(s)"] = Dict("mean" => mean(vals), "target" => s^2, "mcse" => mcse)
        @printf("gate 7 line 1, s = %.2f: mean %.3e, target %.3e, 3 MC SE %.3e\n", s, mean(vals), s^2, 3mcse)
        abs(mean(vals) - s^2) <= 3mcse || gate_fail(7, "estimator mean off at s = $s")
    end
    say("gate 7 line 1 done")
    ub2 = yearly_rel_means(Wd .+ (0.02 / 252) .* zg', yr)
    u1 = ub2[outer_s2[1], :]
    N1 = noise_iid(u1)
    q = [vobs(u1[rand(rng, 1:99, 99), :]) - 2N1 for _ in 1:500]
    mcse = std(q) / sqrt(500)
    g7["line2"] = Dict("mean" => mean(q), "target" => vobs(u1) - N1, "mcse" => mcse)
    @printf("gate 7 line 2: recentred mean %.3e, target %.3e, 3 MC SE %.3e\n", mean(q), vobs(u1) - N1, 3mcse)
    abs(mean(q) - (vobs(u1) - N1)) <= 3mcse || gate_fail(7, "recentring off")
    say("gate 7 line 2 done")
    cov_pct = 0; cov_basic = 0
    for idx in outer_s2
        uo = ub2[idx, :]
        L, U = interval_iid(uo, rng)
        sh = sqrt(max(0.0, est(uo)))
        cov_pct += (L <= 0.02 <= U)
        cov_basic += (2sh - U <= 0.02 <= 2sh - L)
    end
    g7["line3_coverage_percentile"] = cov_pct / 500
    g7["line3_coverage_basic"] = cov_basic / 500
    @printf("gate 7 line 3: coverage percentile %.3f, basic %.3f\n", cov_pct / 500, cov_basic / 500)
    cov_pct / 500 >= 0.80 || gate_fail(7, "coverage $(cov_pct / 500) below 0.80 (plan v6: one-sided)")
    gates["7_estimator"] = g7
    say("gate 7 pass")
end

# ---- data gate count, before any arm is fit
schemes = Dict("primary" => primary, "contiguous" => contig)
function gaps_count()
    n = 0; tot = 0
    for (sc, hs) in schemes, h in hs
        rtr, rref = rows(train_years(h)), rows(fit_years(h))
        for p in patterns_of(sc), (j, s) in enumerate(S_GRID)
            (j == 1 && p != "dense") && continue
            for rr in (rtr, rref)
                g = top_gap(vec(mean(world_rows(h, rr, s, zof(h, p)), dims = 1)))
                tot += 1; n += (g <= GAP_MIN)
            end
        end
    end
    return n, tot
end
ngap, ntot = gaps_count()
gates["data_gate_blocks_at_or_below_GAP_MIN"] = ngap
say("data gate: $ngap of $ntot blocks at or below GAP_MIN")
GATES_ONLY && (println("--gates-only: stopping before any arm is fit"); exit(0))

# ---- ELL with the retry rule
function solve_ell_retry(f::Fit, μ, κ)
    n = length(μ)
    m = Model(Clarabel.Optimizer); set_silent(m)
    set_attribute(m, "max_iter", 1000)
    @variable(m, x[1:n] >= 0)
    @constraint(m, sum(x) == 1)
    if κ == 0
        @objective(m, Max, sum(SCALE * μ[i] * x[i] for i in 1:n))
    else
        @variable(m, t >= 0)
        @constraint(m, vcat(t, f.Lsc' * x) in SecondOrderCone())
        @objective(m, Max, sum(SCALE * μ[i] * x[i] for i in 1:n) - κ * t)
    end
    optimize!(m)
    check_status(m, :clarabel, "ell retry κ=$κ")
    return value.(x)
end
is_status_error(e) = occursin("solver status", sprint(showerror, e))
function ell_book(f, κ, ctx, failed)
    try
        return clean(build_ell(f, κ; ctx = ctx))
    catch e
        is_status_error(e) || rethrow()
        try
            return clean(solve_ell_retry(f, f.mu, κ))
        catch e2
            is_status_error(e2) || rethrow()
            failed[] = true
            return nothing
        end
    end
end
function select_ell(ftr, fref, Rva, ctx, failed)
    X = [ell_book(ftr, k, ctx, failed) for k in GRID_K]
    s = [x === nothing ? -Inf : sharpe(Rva * x) for x in X]
    if all(isinf, s)
        return (h = 0.0, x = clean(solve_robust(fref.mu, fref.d, 0; ctx = ctx)[1]), fallback = true)
    end
    mx = maximum(s)
    i = findfirst(j -> abs(s[j] - mx) < tie_tol, eachindex(GRID_K))
    return (h = GRID_K[i], x = ell_book(fref, GRID_K[i], ctx, failed), fallback = false)
end

# ---- one history, one s, one pattern
function run_one(h, s, z)
    rtr, rva, rte = rows(train_years(h)), rows(val_years(h)), rows(test_years(h))
    rref = vcat(rtr, rva)
    Wtr, Wva, Wte, Wref = (world_rows(h, rr, s, z) for rr in (rtr, rva, rte, rref))
    ftr, fref = fit_block(Wtr, dy[rtr]), fit_block(Wref, dy[rref])
    ctx = "$(h.label) s=$s"
    G, _, _ = select_h((f, g) -> build_box(f, g; ctx = ctx), GRID_G, ftr, Wva)
    failed = Ref(false)
    el = select_ell(ftr, fref, Wva, ctx, failed)
    books = Dict("NOM" => clean(solve_robust(fref.mu, fref.d, 0; ctx = ctx)[1]),
                 "EW" => fill(1 / 30, 30),
                 "BOX" => clean(build_box(fref, G; ctx = ctx)),
                 "ELL" => el.x)
    out = Dict{String,Any}("ell_failed" => failed[], "ell_fallback" => el.fallback,
                           "h" => Dict("BOX" => G, "ELL" => el.h))
    for (a, x) in books
        if x === nothing
            out[a] = nothing
        else
            r = Wte * x
            out[a] = Dict("sharpe" => sharpe(r), "maxdd" => max_drawdown(r), "holdings" => count(x .> held_thresh),
                          "sum" => sum(x), "nan" => any(isnan, x))
        end
    end
    return out
end

# ---- leakage check, s = 0
function leak_corr(hs)
    a, b, c, d = Float64[], Float64[], Float64[], Float64[]
    for h in hs
        rel(rr) = (m = vec(mean(world_rows(h, rr, 0.0, zeros(30)), dims = 1)); m .- mean(m))
        append!(a, rel(rows(fit_years(h)))); append!(b, rel(rows(test_years(h))))
        append!(c, rel(rows(train_years(h)))); append!(d, rel(rows(val_years(h))))
    end
    return cor(a, b), cor(c, d)
end
leak = Dict{String,Any}()
leak["primary_fit_test"], leak["primary_train_val"] = leak_corr(primary)
leak["contiguous_fit_test"], leak["contiguous_train_val"] = leak_corr(contig)
leak["contiguous_by_decade"] = Dict(string(dec) => leak_corr([h for h in contig if h.years[1] ÷ 10 * 10 == dec])[1]
                                    for dec in 1920:10:2010)
println("leakage check: ", leak)

# ---- all arms
results = Dict{String,Any}()
t0 = time()
for sc in ("primary", "contiguous")
    hs = schemes[sc]
    res = Dict{String,Any}(p => [Vector{Any}(undef, length(hs)) for _ in 1:NS] for p in patterns_of(sc))
    for (i, h) in enumerate(hs)
        base = run_one(h, 0.0, zeros(30))
        for p in patterns_of(sc)
            res[p][1][i] = base
            z = zof(h, p)
            for j in 2:NS
                res[p][j][i] = run_one(h, S_GRID[j], z)
            end
        end
        i % 20 == 0 && @printf("%s: %d of %d histories, %.0f s\n", sc, i, length(hs), time() - t0)
    end
    results[sc] = res
end

# ---- gate 8: sanity
for (sc, res) in results, (p, cells) in res, j in 1:NS
    length(cells[j]) == length(schemes[sc]) || gate_fail(8, "missing histories $sc $p $j")
    for o in cells[j], a in ("NOM", "EW", "BOX", "ELL")
        o[a] === nothing && (a == "ELL" && o["ell_failed"]) && continue
        o[a] === nothing && gate_fail(8, "missing book $a $sc $p $j")
        abs(o[a]["sum"] - 1) < 1e-9 || gate_fail(8, "weights of $a sum to $(o[a]["sum"])")
        (o[a]["nan"] || isnan(o[a]["sharpe"])) && gate_fail(8, "NaN in $a $sc $p $j")
    end
end
gates["8_sanity"] = "pass"
say("gate 8 pass; Clarabel ALMOST_OPTIMAL solves: $(length(ALMOST_LOG))")

# ---- s_hat on the real data and its sensitivities
Random.seed!(2031)
s_hat = sqrt(max(0.0, est(ubar)))
L, U = interval_iid(ubar, Random.default_rng())
lag1 = mean([cor(ubar[1:98, i], ubar[2:99, i]) for i in 1:30])
Random.seed!(2032)
blk_mts = Vector{Vector{Float64}}(); blk_v = Float64[]
for b in 1:1000
    st = rand(1:99, 20)
    idx = reduce(vcat, [mod1.(s0:s0+4, 99) for s0 in st])[1:99]
    ub = ubar[idx, :]
    push!(blk_mts, vec(mean(ub, dims = 1))); push!(blk_v, vobs(ub))
end
N_blk = mean([var([m[i] for m in blk_mts]) for i in 1:30])
s_hat_blk = sqrt(max(0.0, vobs(ubar) - N_blk))
sb = [sqrt(max(0.0, v - 2N_blk)) for v in blk_v]
L_blk, U_blk = pct(sb, 0.05), pct(sb, 0.95)
realdat = Dict("s_hat" => s_hat, "L" => L, "U" => U, "V_obs" => vobs(ubar), "N" => noise_iid(ubar),
            "lag1_autocorr_mean" => lag1, "s_hat_blk" => s_hat_blk, "L_blk" => L_blk, "U_blk" => U_blk,
            "N_blk" => N_blk, "z_real" => z_real)
@printf("s_hat %.4f [%.4f, %.4f]; block %.4f [%.4f, %.4f]; lag-1 autocorrelation %.3f\n",
        s_hat, L, U, s_hat_blk, L_blk, U_blk, lag1)

summary = Dict{String,Any}(
    "header" => Dict("code_commit" => code_commit,
                     "run_date" => string(Dates.now()), "julia" => string(VERSION), "packages" => pkgver,
                     "seeds" => [2027, 2028, 2029, 2030, 2031, 2032]),
    "s_grid" => S_GRID, "kappa_grid" => GRID_K, "gates" => gates, "almost_optimal" => length(ALMOST_LOG),
    "leakage" => leak, "headroom" => headroom, "real" => realdat,
    "histories" => Dict(sc => [Dict("label" => h.label, "years" => h.years, "r" => h.r, "k" => h.k) for h in schemes[sc]]
                        for sc in keys(schemes)),
    "results" => results)
open(joinpath(ROOT, "results", "injection_summary.json"), "w") do io
    write(io, JSON.json(summary))
end
println("wrote results/injection_summary.json in $(round(time() - t0)) s")
