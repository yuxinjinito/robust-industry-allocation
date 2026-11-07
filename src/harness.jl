# Harness for the extension arms. The definitions marked "baseline" are copied from
# baseline/nb.jl unchanged, so the R-gate can reproduce the report through this file.

using JuMP, HiGHS, Clarabel, Statistics, LinearAlgebra, Random, Printf
const MOI = JuMP.MOI

# ---- baseline: data loading (nb.jl cell 1), unchanged
function load_vw_daily(path)
    lines = readlines(path)
    i0 = findfirst(l -> occursin(r"^\s*\d{8},", l), lines)
    industries = String.(strip.(split(lines[i0-1], ','))[2:end])
    dates = Int[]
    rows = Vector{Vector{Float64}}()
    for l in lines[i0:end]
        m = match(r"^\s*(\d{8}),(.*)$", l)
        m === nothing && break
        push!(dates, parse(Int, m.captures[1]))
        push!(rows, parse.(Float64, split(m.captures[2], ',')))
    end
    R = reduce(vcat, transpose.(rows)) ./ 100
    return dates, industries, R
end

# ---- baseline: inputs, scoring and selection constants (nb.jl cells 9-10), unchanged
win(dv, lo, hi) = (dv .>= lo) .& (dv .<= hi)
fit_inputs(Rw) = (vec(mean(Rw, dims=1)), 2 .* vec(std(Rw, dims=1)) ./ sqrt(size(Rw, 1)))
clean(x) = (y = max.(x, 0.0); y ./ sum(y))
sharpe(r) = sqrt(252) * mean(r) / std(r)
ann(r) = 100 * 252 * r
ann_ret(r) = 100 * 252 * mean(r)
ann_vol(r) = 100 * sqrt(252) * std(r)
function max_drawdown(r)
    W = vcat(1.0, cumprod(1 .+ r))
    P = accumulate(max, W)
    return 100 * maximum((P .- W) ./ P)
end
const held_thresh = 1e-4
const tie_tol = 1e-4

# ---- solver status. HiGHS must report OPTIMAL. Clarabel may report ALMOST_OPTIMAL,
# which is accepted and logged; `strict` (used by E1) refuses it.
const ALMOST_LOG = String[]
function check_status(m, solver, ctx; strict = false)
    st = termination_status(m)
    st == MOI.OPTIMAL && return
    if solver == :clarabel && !strict && st == MOI.ALMOST_OPTIMAL
        push!(ALMOST_LOG, ctx)
        return
    end
    error("solver status $st ($solver) at $ctx")
end

# ---- baseline box LP (nb.jl cell 3), with the status check added. The check reads the
# status only, so the returned values are those of the baseline.
function solve_robust(μ, d, Γ; ctx = "")
    n = length(μ)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x[1:n] >= 0)
    @variable(m, z >= 0)
    @variable(m, p[1:n] >= 0)
    @constraint(m, sum(x) == 1)
    @constraint(m, dualfeas[i in 1:n], z + p[i] >= d[i] * x[i])
    @objective(m, Max, sum(μ[i] * x[i] for i in 1:n) - Γ * z - sum(p))
    optimize!(m)
    check_status(m, :highs, "box Γ=$Γ $ctx")
    return value.(x), objective_value(m), value(z)
end

# ---- everything a fit block provides to any arm
const SCALE = 1e4    # basis points per day for the conic solves

struct Fit
    mu::Vector{Float64}       # sample mean
    d::Vector{Float64}        # 2 SE
    Lsc::Matrix{Float64}      # SCALE * chol(S / T).L, so ||Lsc' x|| = SCALE * sqrt(x' Ω x)
    Omega::Matrix{Float64}    # S / T
    beta::Vector{Float64}     # on the equal-weight market proxy
    mbar::Float64             # mean of the proxy
    dcl::Vector{Float64}      # 2 SE_cl, year-clustered
    drift_ratio::Float64      # median_i SE_cl / SE_iid
    wJ::Float64               # Jorion's formula value, descriptive
    T::Int
end

# year-clustered SE of the mean: sd over calendar years of the yearly means / sqrt(Y)
function se_cluster(Rw, years)
    ys = unique(years)
    M = reduce(vcat, [mean(Rw[years .== y, :], dims=1) for y in ys])
    return vec(std(M, dims=1)) ./ sqrt(length(ys))
end

function fit_block(Rw, dv)
    T, N = size(Rw)
    mu, d = fit_inputs(Rw)
    S = cov(Rw)
    Omega = S ./ T
    Lsc = SCALE .* Matrix(cholesky(Symmetric(Omega)).L)
    m = vec(mean(Rw, dims=2))
    beta = [cov(Rw[:, i], m) / var(m) for i in 1:N]
    mbar = mean(m)
    years = dv .÷ 10000
    secl = se_cluster(Rw, years)
    seiid = d ./ 2
    Si = inv(S)
    one = ones(N)
    mu0 = (one' * Si * mu) / (one' * Si * one)
    P = (T - N - 2) / (T - 1) .* Si
    q = (mu .- mu0)' * P * (mu .- mu0)
    wJ = (N + 2) / ((N + 2) + T * q)
    return Fit(mu, d, Lsc, Omega, beta, mbar, 2 .* secl, median(secl ./ seiid), wJ, T)
end

# mean of the drift reference: the same median ratio after permuting rows, dates fixed
function shuffled_drift_ratio(Rw, dv, rng; nperm = 200)
    years = dv .÷ 10000
    seiid = vec(std(Rw, dims=1)) ./ sqrt(size(Rw, 1))
    vals = Float64[]
    for _ in 1:nperm
        Rp = Rw[randperm(rng, size(Rw, 1)), :]
        push!(vals, median(se_cluster(Rp, years) ./ seiid))
    end
    return mean(vals)
end

shrink(f::Fit, w) = (1 - w) .* f.mu .+ w .* (f.beta .* f.mbar)

# ---- ellipsoid robust counterpart, Clarabel, scaled inputs
function solve_ell(f::Fit, μ, κ; ctx = "", strict = false)
    n = length(μ)
    m = Model(Clarabel.Optimizer)
    set_silent(m)
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
    check_status(m, :clarabel, "ell κ=$κ $ctx"; strict = strict)
    return value.(x)
end

# ---- mean-variance QP on the same scaled inputs, HiGHS, for E1
function solve_qp(f::Fit, μ, λ)
    n = length(μ)
    Os = SCALE^2 .* f.Omega
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x[1:n] >= 0)
    @constraint(m, sum(x) == 1)
    @objective(m, Max, sum(SCALE * μ[i] * x[i] for i in 1:n) - λ * (x' * Os * x))
    optimize!(m)
    check_status(m, :highs, "qp λ=$λ")
    return value.(x)
end

# ---- grids
const GRID_G = collect(0:30)
const GRID_K = vcat(0.0, [2 * 10^(j / 16) for j in -20:9])
const GRID_W = collect(0:30) ./ 30

# ---- the one selection rule, used by the holdout and the rolling loop alike
function select_h(build, grid, ftr::Fit, Rva)
    X = [clean(build(ftr, h)) for h in grid]
    s = [sharpe(Rva * x) for x in X]
    mx = maximum(s)
    i = findfirst(j -> abs(s[j] - mx) < tie_tol, eachindex(grid))   # grid ascending: smallest
    return grid[i], X, s
end

top_gap(v) = (s = sort(v, rev = true); s[1] - s[2])
const GAP_MIN = 1e-8

function gate_gap(v, ctx)
    g = top_gap(v)
    g > GAP_MIN || error("GATE 3 FAILED: data gate, top-two gap $g <= $GAP_MIN at $ctx")
    return g
end

const ARMS = [:NOM, :EW, :BOX, :ELL, :RBOX, :RELL, :DRIFT]

# named builds, shared by run_arms and the R-gate
build_box(f::Fit, G; ctx = "") = solve_robust(f.mu, f.d, G; ctx = ctx)[1]
build_ell(f::Fit, κ; ctx = "") = solve_ell(f, f.mu, κ; ctx = ctx)
build_drift(f::Fit, G; ctx = "") = solve_robust(f.mu, f.dcl, G; ctx = ctx)[1]

# the report's windows: holdout, and the rolling rounds y0 in 1926:3:2016
holdout_masks(dates) = (tr = win(dates, 20150101, 20201231), va = win(dates, 20210101, 20221231),
                        te = win(dates, 20230101, 20251231))
function rolling_windows(dates)
    out = []
    for y0 in 1926:3:2016
        tr = win(dates, y0 * 10000 + 101, (y0 + 5) * 10000 + 1231)
        va = win(dates, (y0 + 6) * 10000 + 101, (y0 + 7) * 10000 + 1231)
        te = win(dates, (y0 + 8) * 10000 + 101, (y0 + 10) * 10000 + 1231)
        (sum(tr) < 1000 || sum(va) < 300 || sum(te) < 500) && continue
        push!(out, (y0 = y0 + 8, tr = tr, va = va, te = te))
    end
    return out
end

# fits for one window, from masks
function window_fits(dates, R, w)
    ref = w.tr .| w.va
    return fit_block(R[w.tr, :], dates[w.tr]), fit_block(R[ref, :], dates[ref])
end

# Selects and freezes every requested arm on one window. ftr is the train fit, fref the
# train+validation refit, Rva the validation returns. Returns arm => (h, book, extra).
function run_arms(ftr::Fit, fref::Fit, Rva; arms = ARMS, ctx = "")
    out = Dict{Symbol,Any}()
    for f in (ftr, fref)
        gate_gap(f.mu, "mu_hat $ctx")
    end
    if :NOM in arms
        out[:NOM] = (h = nothing, x = clean(solve_robust(fref.mu, fref.d, 0; ctx = ctx)[1]))
    end
    if :EW in arms
        out[:EW] = (h = nothing, x = fill(1 / 30, 30))
    end
    if :BOX in arms || :RBOX in arms
        G, _, _ = select_h((f, G) -> build_box(f, G; ctx = ctx), GRID_G, ftr, Rva)
        out[:BOX] = (h = G, x = clean(build_box(fref, G; ctx = ctx)))
    end
    if :ELL in arms || :RELL in arms
        κ, _, _ = select_h((f, k) -> build_ell(f, k; ctx = ctx), GRID_K, ftr, Rva)
        out[:ELL] = (h = κ, x = clean(build_ell(fref, κ; ctx = ctx)))
    end
    if :RBOX in arms
        G = out[:BOX].h
        if G == 0
            for f in (ftr, fref), w in GRID_W
                gate_gap(shrink(f, w), "mu_w w=$w RBOX $ctx")
            end
        end
        w, _, _ = select_h((f, w) -> solve_robust(shrink(f, w), f.d, G; ctx = ctx)[1], GRID_W, ftr, Rva)
        out[:RBOX] = (h = w, x = clean(solve_robust(shrink(fref, w), fref.d, G; ctx = ctx)[1]), fixed = G)
    end
    if :RELL in arms
        κ = out[:ELL].h
        if κ == 0
            for f in (ftr, fref), w in GRID_W
                gate_gap(shrink(f, w), "mu_w w=$w RELL $ctx")
            end
        end
        w, _, _ = select_h((f, w) -> solve_ell(f, shrink(f, w), κ; ctx = ctx), GRID_W, ftr, Rva)
        out[:RELL] = (h = w, x = clean(solve_ell(fref, shrink(fref, w), κ; ctx = ctx)), fixed = κ)
    end
    if :DRIFT in arms
        G, _, _ = select_h((f, G) -> build_drift(f, G; ctx = ctx), GRID_G, ftr, Rva)
        out[:DRIFT] = (h = G, x = clean(build_drift(fref, G; ctx = ctx)))
    end
    return Dict(a => out[a] for a in arms)
end

# ---- scoring
const COSTS_BPS = [5, 10, 20, 50]

# TO_t: total trade (buys plus sells) to restore x after day t
function turnover(Rte, x)
    r = Rte * x
    drift = (Rte .+ 1) .* x' ./ (1 .+ r)
    return vec(sum(abs.(x' .- drift), dims = 2))
end

# monthly reset at the close of the last trading day of each month, not on the last test day
function monthly_net(Rte, dte, x, c)
    n = size(Rte, 1)
    w = copy(x)
    rn = zeros(n)
    for t in 1:n
        rt = dot(w, Rte[t, :])
        wd = w .* (1 .+ Rte[t, :]) ./ (1 + rt)
        if t < n && (dte[t] ÷ 100) != (dte[t+1] ÷ 100)
            rn[t] = rt - c * sum(abs.(x .- wd))
            w = copy(x)
        else
            rn[t] = rt
            w = wd
        end
    end
    return rn
end

function score(Rte, dte, x)
    r = Rte * x
    TO = turnover(Rte, x)
    sm = Dict{String,Any}(
        "ann_ret" => ann_ret(r), "ann_vol" => ann_vol(r), "sharpe" => sharpe(r),
        "maxdd" => max_drawdown(r), "holdings" => count(x .> held_thresh),
        "turnover_annual" => 252 * mean(TO))
    for c in COSTS_BPS
        sm["sharpe_net_$(c)bps"] = sharpe(r .- (c / 1e4) .* TO)
    end
    sm["sharpe_net_10bps_monthly"] = sharpe(monthly_net(Rte, dte, x, 10 / 1e4))
    # near-zero weights, to tell real holdings from interior-point dust
    below = filter(<=(held_thresh), x); above = filter(>(held_thresh), x)
    sm["largest_below_thresh"] = isempty(below) ? 0.0 : maximum(below)
    sm["smallest_above_thresh"] = isempty(above) ? 0.0 : minimum(above)
    sm["has_weight_1e-5_to_1e-3"] = any(1e-5 .<= x .<= 1e-3)
    return sm
end
