println("\n#### CELL 1")
using JuMP, HiGHS, Statistics, Plots
gr()

# read the value-weighted daily table out of the Ken French CSV. the file stacks four
# tables on the same 30 industry columns (value-weighted returns, then equal-weighted,
# then two of firm counts), and the only thing marking a data row is that it starts with
# an 8-digit date. so find the first such row, take the line above it as the header, and
# stop at the first line that is not a data row, which is where the next table begins.
function load_vw_daily(path)
    lines = readlines(path)
    i0 = findfirst(l -> occursin(r"^\s*\d{8},", l), lines)       # first data row of the first table
    industries = String.(strip.(split(lines[i0-1], ','))[2:end]) # header row holds industry names
    dates = Int[]
    rows = Vector{Vector{Float64}}()
    for l in lines[i0:end]
        m = match(r"^\s*(\d{8}),(.*)$", l)   # the date and the 30 numbers, captured separately
        m === nothing && break               # first non-data line: the next table starts here
        push!(dates, parse(Int, m.captures[1]))
        push!(rows, parse.(Float64, split(m.captures[2], ',')))
    end
    # the file quotes percent (0.51 means +0.51% on the day), every formula below wants
    # decimals, so convert once here instead of in each caller
    R = reduce(vcat, transpose.(rows)) ./ 100
    return dates, industries, R
end

# the CSV snapshot sits in the same folder as this notebook. it is a frozen copy of the
# May 2026 build, since the library revises recent history every month.
dates, industries, R = load_vw_daily("30_Industry_Portfolios_Daily.csv")

# R now runs from 1926 to 2026; this mask picks the 2015-2025 rows we use. the three
# checks fail loudly rather than letting a parsing mistake reach the model.
sel_all = (dates .>= 20150101) .& (dates .<= 20251231)
@assert length(industries) == 30      # 30 columns means we read an industry table, not a count table
@assert sum(sel_all) == 2766          # the trading-day count we checked separately when choosing the data
@assert all(R[sel_all, :] .> -0.98)   # French codes a missing observation as -99.99, which is -0.9999
                                      # after the divide. no industry index drops 98% in one day, so
                                      # this catches missing codes without catching real returns.
println("data checks passed")

println("\n#### CELL 2")
# ---- the two inputs the model needs, both from the estimation window only
sel_est = (dates .>= 20150101) .& (dates .<= 20221231)   # 2023-2025 is in R but is not read here
Rest = R[sel_est, :]
T = size(Rest, 1)                            # trading days in the estimation window

μ = vec(mean(Rest, dims=1))                  # sample mean daily return, what a nominal optimizer would trust

# d_i is how far that mean could plausibly be off. the standard error of a mean is
# std/sqrt(T), and we take twice it, roughly a 95% half-width. it is the amount the
# adversary may subtract from any one industry, so it sets how strong the adversary is.
d = 2 .* vec(std(Rest, dims=1)) ./ sqrt(T)

ann(r) = 100 * 252 * r                       # daily decimal -> annualized percent, for printing only

println("annualized mean returns: min ", round(minimum(ann.(μ)), digits=1),
        "%, max ", round(maximum(ann.(μ)), digits=1), "%")
println("annualized 2xSE:         min ", round(minimum(ann.(d)), digits=1),
        "%, max ", round(maximum(ann.(d)), digits=1), "%")

# the model's probabilistic guarantee assumes the 30 estimation errors are independent,
# and industries move together, so measure it here. subtracting 30 drops the diagonal of
# ones and 30*29 counts the remaining ordered pairs.
C = cor(Rest)
println("average pairwise correlation: ", round((sum(C) - 30) / (30 * 29), digits=2))

# the premise in one picture: thirty estimated means and their ±d bands on one axis,
# sorted by mean so the overlap is visible at a glance
ord = sortperm(μ, rev=true)
p_in = scatter(1:30, ann.(μ[ord]), yerror = ann.(d[ord]), legend = false,
               xticks = (1:30, industries[ord]), xrotation = 90, size = (900, 430),
               bottom_margin = 10Plots.mm, left_margin = 10Plots.mm,
               ylabel = "Annualized mean return (%)",
               title = "Estimated means with ±d error bars, 2015-2022")
savefig(p_in, "fig_inputs.png")
p_in

println("\n#### CELL 3")
# ---- the robust model as a single LP
# the two-level game (we choose x, then an adversary lowers the estimated mean of up to Γ
# industries by d_i each) cannot go to a solver directly, but the adversary's inner problem
# is itself an LP, so replacing it by its dual collapses the game into the one LP below.
# derivation in Section 2. returns the weights, the guaranteed worst-case value, and z.
function solve_robust(μ, d, Γ)
    n = length(μ)
    m = Model(HiGHS.Optimizer)
    set_silent(m)                   # the solver's own log is not part of the report

    @variable(m, x[1:n] >= 0)       # portfolio weights. nonnegative means no short selling
    @variable(m, z >= 0)            # dual of the adversary's budget constraint sum(r_i) <= Γ
    @variable(m, p[1:n] >= 0)       # duals of the adversary's per-asset caps r_i <= 1

    # all capital is invested: the weights are shares of one dollar and must exhaust it
    @constraint(m, sum(x) == 1)

    # dual feasibility, one row per industry. whatever the adversary could take from
    # industry i, d_i * x_i, has to be covered by the budget price z plus the per-asset
    # price p_i, so this is what makes z and p pay for the damage.
    @constraint(m, dualfeas[i in 1:n], z + p[i] >= d[i] * x[i])

    # estimated return minus what the adversary can remove, priced through the duals.
    # maximizing this is exactly maximizing the worst case.
    @objective(m, Max, sum(μ[i] * x[i] for i in 1:n) - Γ * z - sum(p))

    optimize!(m)
    return value.(x), objective_value(m), value(z)
end;   # the semicolon keeps IJulia from echoing the function object under the cell

println("\n#### CELL 4")
# ---- validating the reformulation
# the dualization above is the one step where a silent algebra slip gives a model that
# still runs and is wrong everywhere. following the instructor's suggestion on the
# proposal, we check it on a five-industry toy against two methods that use no duality at
# all. all three have to agree.

# check 1: for fixed weights the adversary just hits the Γ largest d_i * x_i,
# so the worst case can be computed by sorting
worst_direct(x, μ, d, Γ) = sum(μ .* x) - sum(sort(d .* x, rev=true)[1:Γ])

# all ways to choose k industries out of 1:n, built recursively: pick the largest member
# `last`, then choose the other k-1 below it. affordable only because the toy has 5 assets.
function subsets_of_size(n, k)
    k == 0 && return [Int[]]
    result = Vector{Vector{Int}}()
    for last in k:n
        for head in subsets_of_size(last - 1, k - 1)
            push!(result, vcat(head, last))
        end
    end
    return result
end

# check 2: re-solve the whole two-level problem without duality. t stands for the value
# after the worst attack, and one constraint per attack subset holds t below the value
# under that subset, so maximizing t optimizes against every attack at once.
function solve_enum(μ, d, Γ)
    n = length(μ)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, x[1:n] >= 0)                        # portfolio weights, no short selling
    @variable(m, t)                                  # the worst case over all attacks, free in sign
    @constraint(m, sum(x) == 1)                      # all capital invested
    for S in subsets_of_size(n, Γ)                   # one row per attack subset S of size Γ:
        @constraint(m, t <= sum(μ[i] * x[i] for i in 1:n) - sum(d[i] * x[i] for i in S; init = 0.0))
    end                                              # t is no larger than the value under S
    @objective(m, Max, t)
    optimize!(m)
    return value.(x), objective_value(m)
end

# run all three on the first five industries, at every budget a five-asset problem allows
ntoy = 5
μ_toy, d_toy = μ[1:ntoy], d[1:ntoy]
println("toy industries: ", join(industries[1:ntoy], ", "))
for Γ in 0:ntoy
    x1, o1, _ = solve_robust(μ_toy, d_toy, Γ)        # our dual reformulation
    x2, o2 = solve_enum(μ_toy, d_toy, Γ)             # brute force, no duality
    w1 = worst_direct(x1, μ_toy, d_toy, Γ)           # direct sort, no optimization
    diff = max(abs(o1 - o2), abs(o1 - w1))
    println("Γ = ", Γ, ": dual LP = ", round(o1, sigdigits=6),
            ", enumeration LP = ", round(o2, sigdigits=6),
            ", direct sort = ", round(w1, sigdigits=6),
            ", max diff = ", round(diff, sigdigits=3))
    @assert diff < 1e-9   # the three share no code path, so this fails loudly if the model breaks
end
println("toy validation passed")

println("\n#### CELL 5")
# ---- the full problem at four budget levels
# Γ = 0 trusts the estimates completely, Γ = 30 assumes every one of them could be wrong,
# and 5 and 15 sit in between. the largest holdings show how the portfolio changes as the
# model is asked to trust the data less.
println("full model on all 30 industries:")
for Γ in (0, 5, 15, 30)
    x, obj, z = solve_robust(μ, d, Γ)
    # a weight the solver reports as 1e-12 is vertex dust, not a position, so an industry
    # counts as held only above 1e-4
    holdings = [(industries[i], 100 * x[i]) for i in 1:30 if x[i] > 1e-4]
    sort!(holdings, by = h -> -h[2])   # largest positions first, so the top five really is the top five
    top5 = join([h[1] * " " * string(round(h[2], digits=1)) * "%" for h in holdings[1:min(5, end)]], ", ")
    println("Γ = ", Γ, ": worst-case return ", round(ann(obj), digits=2), "%, ",
            length(holdings), " holdings; largest: ", top5)
end

println("\n#### CELL 6")
# ---- the whole robustness path at once
# solve at every integer budget from 0 to 30 and keep three series: the guaranteed
# worst-case value (the price-of-robustness curve), the dual z, and how many industries
# are held. each solve is a 61-variable LP and takes milliseconds.
Γ_range = 0:30
worst_case = Float64[]
z_values = Float64[]
num_holdings = Int[]
for Γ in Γ_range
    x, obj, z = solve_robust(μ, d, Γ)
    push!(worst_case, obj)
    push!(z_values, z)
    push!(num_holdings, count(x .> 1e-4))   # same dust threshold as above
end

# z is the dual on the adversary's budget constraint, so it is that budget's shadow price:
# one more unit of Γ should cost us exactly z, and the curve's slope must equal -z*. a
# finite difference of the curve checks the theory against the solver's own output, and
# the two agree wherever the curve is straight between the two Γ being differenced.
println("slope of the curve vs -z* (equal on linear stretches):")
for k in 1:10
    println("Γ = ", Γ_range[k], ": slope = ", round(worst_case[k+1] - worst_case[k], sigdigits=4),
            ", -z* = ", round(-z_values[k], sigdigits=4))
end

println("\n#### CELL 7")
# the price of robustness: how much guaranteed return is given up for each extra industry
# the adversary may attack. annualized here for reading; the model works in daily decimals.
p_price = plot(Γ_range, ann.(worst_case), marker = :circle, legend = false,
     xlabel = "Γ (assets the adversary may affect)",
     ylabel = "Worst-case annualized return (%)",
     title = "Price of robustness curve")
savefig(p_price, "fig_price_curve.png")   # saved under the name section 4 links, so the report rebuilds from the notebook alone
p_price

println("\n#### CELL 8")
# how many industries the optimal portfolio holds along the same path. plotted separately
# from the curve above because the two answer different questions: what robustness costs,
# and what it does to the shape of the portfolio.
p_hold = plot(Γ_range, num_holdings, marker = :circle, legend = false,
     xlabel = "Γ (assets the adversary may affect)",
     ylabel = "Number of holdings",
     title = "Portfolio size along the robustness path")
savefig(p_hold, "fig_holdings.png")
p_hold

println("\n#### CELL 9")
using Printf

# three disjoint calendar blocks. the upper bound on the holdout matters, since the CSV
# runs past 2025-12-31 and an open-ended mask would pull those rows in.
win(dv, lo, hi) = (dv .>= lo) .& (dv .<= hi)
sel_tr = win(dates, 20150101, 20201231)
sel_va = win(dates, 20210101, 20221231)
sel_ho = win(dates, 20230101, 20251231)
@assert !any(sel_est .& sel_ho)         # no day is used both to fit and to score
@assert (sel_tr .| sel_va) == sel_est   # train and validation partition the estimation window

# section 2's estimator, plus the scoring convention: a weight vector is read as a
# daily-rebalanced constant-weight book, so the day's return is x'R_t.
fit_inputs(Rw) = (vec(mean(Rw, dims=1)), 2 .* vec(std(Rw, dims=1)) ./ sqrt(size(Rw, 1)))
clean(x) = (y = max.(x, 0.0); y ./ sum(y))   # clamp the LP's -0.0 entries, renormalize
sharpe(r) = sqrt(252) * mean(r) / std(r)
held_thresh = 1e-4    # a real holding, against solver dust around 1e-16
tie_tol = 1e-4        # two validation Sharpes closer than this count as tied

# the whole selection as a function of a data block, so the leakage audit further down can
# rerun it on truncated data instead of reimplementing it.
function select_and_fit(dv, Rw)
    mu_f, d_f = fit_inputs(Rw[win(dv, 20150101, 20201231), :])          # fit on the train block
    X = [clean(solve_robust(mu_f, d_f, G)[1]) for G in 0:30]            # one candidate book per Γ
    s = [sharpe(Rw[win(dv, 20210101, 20221231), :] * x) for x in X]     # each scored on validation
    ties = [G for G in 0:30 if abs(s[G+1] - maximum(s)) < tie_tol]      # maximizers, within tolerance
    x_star = clean(solve_robust(fit_inputs(Rw[win(dv, 20150101, 20221231), :])...,
                                minimum(ties))[1])                      # refit on the full window
    return ties, X, s, x_star
end

tied, X_tr, sh_va, x_rob = select_and_fit(dates, R)
Gamma_star = minimum(tied)

# four budgets that carry the shape of the sweep: the trust-everything end, the dip just
# past it, the best score, and the point where the fit falls back to a single industry.
println("validation Sharpe by budget, fitted on 2015-2020 and scored on 2021-2022")
for G in (0, 1, 11, 13)
    @printf("  Γ = %2d: %2d holdings, Sharpe %7.4f\n", G, count(X_tr[G+1] .> held_thresh), sh_va[G+1])
end
@printf("selected Γ* = %d\n", Gamma_star)

println("\n#### CELL 10")
# with no adversary budget the Γ = 0 objective is the unperturbed in-sample mean of the
# single best industry, so that number bounds nothing. only the Γ* row is a real bound.
x_nom = clean(solve_robust(μ, d, 0)[1])
x_ew = fill(1 / 30, 30)
books = [("robust Γ=$Gamma_star", x_rob), ("nominal Γ=0", x_nom), ("equal weight 1/N", x_ew)]
@printf("LP worst-case bound %.2f%% at Γ* = %d, against %.2f%% at Γ = 0\n",
        ann(solve_robust(μ, d, Gamma_star)[2]), Gamma_star, ann(solve_robust(μ, d, 0)[2]))
@printf("  the second is just the annualized in-sample mean of %s, the best single industry\n",
        industries[argmax(μ)])

ann_ret(r) = 100 * 252 * mean(r)
ann_vol(r) = 100 * sqrt(252) * std(r)

# drawdown runs on the wealth path of one dollar. the leading 1.0 is the starting capital:
# without it the running peak starts at day one's wealth and a first-day loss never registers.
function max_drawdown(r)
    W = vcat(1.0, cumprod(1 .+ r))
    P = accumulate(max, W)
    return 100 * maximum((P .- W) ./ P)
end

# the holdout, indexed for the first time
Rho = R[sel_ho, :]
println("holdout 2023-2025     hold  ann_ret%  ann_vol%   sharpe   maxDD%")
for (lab, x) in books
    r = Rho * x
    @printf("  %-18s %3d %9.2f %9.2f %8.4f %8.2f\n", lab, count(x .> held_thresh),
            ann_ret(r), ann_vol(r), sharpe(r), max_drawdown(r))
end

println("\n#### CELL 11")
# an industry leaves the uncertainty set when its realized holdout mean lands below μ̂_i - d_i,
# which is the event the adversary was paid to insure against.
mu_ho = vec(mean(Rho, dims=1))
below = mu_ho .< (μ .- d)
@printf("industries whose 2023-2025 mean fell below μ̂ - d: %d of 30 (%s)\n",
        count(below), join(industries[below], ", "))
@printf("%s, the whole nominal book, ranked %d of 30 on holdout mean\n",
        industries[argmax(μ)], findfirst(==(argmax(μ)), sortperm(mu_ho, rev=true)))

# leakage audit: delete every post-2022 row and run the same selection from scratch on what
# is left. had a holdout row reached μ̂, d or Γ*, the rerun would move Γ* or move a weight.
keep = dates .<= 20221231
tied_c, _, _, x_c = select_and_fit(dates[keep], R[keep, :])
@printf("leakage audit: post-2022 rows deleted, Γ* = %d, largest weight difference %g\n",
        minimum(tied_c), maximum(abs.(x_c .- x_rob)))

println("\n#### CELL 12")
plt_oos = plot(hcat([vcat(1.0, cumprod(1 .+ Rho * x)) for (_, x) in books]...),
               label = permutedims(first.(books)), linestyle = [:solid :solid :dash],
               linewidth = 1.8, legend = :topleft, size = (900, 550), left_margin = 5Plots.mm,
               xlabel = "Holdout trading day (2023-01-03 to 2025-12-31)",
               ylabel = "Value of one dollar",
               title = "Cumulative return of one dollar over the 2023-2025 holdout")
savefig(plt_oos, "fig_holdout.png")
plt_oos

println("\n#### CELL 13")
# the same protocol as above, repeated across a century instead of once. each round fits
# on 6 years, picks Γ* on the next 2, refits on all 8, and scores the following 3. rounds
# start every 3 years from 1926, so no two test windows overlap.
function one_round(y0)
    tr = win(dates, y0 * 10000 + 101, (y0 + 5) * 10000 + 1231)          # 6 years to fit
    va = win(dates, (y0 + 6) * 10000 + 101, (y0 + 7) * 10000 + 1231)    # 2 years to pick Γ*
    te = win(dates, (y0 + 8) * 10000 + 101, (y0 + 10) * 10000 + 1231)   # 3 years to score
    (sum(tr) < 1000 || sum(va) < 300 || sum(te) < 500) && return nothing # incomplete at the edges

    mu_f, d_f = fit_inputs(R[tr, :])
    X = [clean(solve_robust(mu_f, d_f, G)[1]) for G in 0:30]
    s = [sharpe(R[va, :] * x) for x in X]
    G = minimum([g for g in 0:30 if abs(s[g+1] - maximum(s)) < tie_tol])

    mu_e, d_e = fit_inputs(R[tr .| va, :])                              # refit on all 8 years
    rob, nom = clean(solve_robust(mu_e, d_e, G)[1]), clean(solve_robust(mu_e, d_e, 0)[1])
    ew = fill(1 / 30, 30)
    Rte = R[te, :]
    return (y0 = y0 + 8, G = G,
            ret_rob = ann_ret(Rte * rob), ret_nom = ann_ret(Rte * nom),
            ret_ew = ann_ret(Rte * ew),
            dd_rob = max_drawdown(Rte * rob), dd_nom = max_drawdown(Rte * nom),
            dd_ew = max_drawdown(Rte * ew))
end

rounds = filter(!isnothing, [one_round(y) for y in 1926:3:2016])

# where the validation step chose Γ* = 0 the model declined to buy any protection, the two
# books are the same portfolio, and there is nothing to compare. the rest are the rounds
# where robustness was actually paid for.
paid = [r for r in rounds if r.G > 0]
@printf("%d test windows, %d-%d. Γ* = 0 in %d of them, so the model bought protection in %d\n",
        length(rounds), rounds[1].y0, rounds[end].y0 + 2,
        length(rounds) - length(paid), length(paid))

# the nominal book is the controlled comparison: same LP, same inputs, only Γ differs, so
# any gap is attributable to robustness alone. quartiles are shown because the spread
# across rounds is wide and a median alone would hide it.
gap_dd = [r.dd_rob - r.dd_nom for r in paid]
gap_ret = [r.ret_rob - r.ret_nom for r in paid]
println("\nrobust minus nominal, so a negative drawdown gap and a positive return gap favour robust")
println("                     robust better     median   middle half")
@printf("  maximum drawdown %10d of %-3d %8.1f pp %6.1f to %5.1f\n", count(<(0), gap_dd),
        length(paid), median(gap_dd), quantile(gap_dd, .25), quantile(gap_dd, .75))
@printf("  annual return    %10d of %-3d %8.1f pp %6.1f to %5.1f\n", count(>(0), gap_ret),
        length(paid), median(gap_ret), quantile(gap_ret, .25), quantile(gap_ret, .75))

# a test window counts as stressed if an equal-weight book, which knows nothing about our
# model, drew down more than 20% in it. the rule uses the market, not our own results.
st = [r for r in paid if r.dd_ew > 20]
ca = [r for r in paid if r.dd_ew <= 20]
@printf("\ndrawdown gap by market condition: %.1f pp over the %d stressed rounds, %.1f pp over the %d calm ones\n",
        median([r.dd_rob - r.dd_nom for r in st]), length(st),
        median([r.dd_rob - r.dd_nom for r in ca]), length(ca))

# the same rounds scored against equal weight instead, which bounds what the LP can claim:
# beating a single-industry book only shows that diversifying helps.
gap_ew = [r.dd_rob - r.dd_ew for r in paid]
@printf("against equal weight instead: smaller drawdown in %d of %d rounds, median gap %.1f pp\n",
        count(<(0), gap_ew), length(paid), median(gap_ew))

# the 22 gaps as one picture. bars below zero are rounds the robust book lost less, and
# the coloring reuses the stress rule above.
xs = string.(getfield.(paid, :y0))
g_st = [paid[i].dd_ew > 20 ? gap_dd[i] : NaN for i in eachindex(paid)]
g_ca = [paid[i].dd_ew > 20 ? NaN : gap_dd[i] for i in eachindex(paid)]
p_roll = bar(xs, g_st, label = "stressed window", color = :firebrick,
             xrotation = 90, size = (900, 430), bottom_margin = 8Plots.mm,
             left_margin = 10Plots.mm, xticks = (1:length(xs), xs),
             ylims = (minimum(gap_dd) - 3, maximum(gap_dd) + 3),
             xlabel = "Test window start year",
             ylabel = "Drawdown gap, robust minus nominal (pp)",
             title = "Drawdown gap in each paid round")
bar!(p_roll, xs, g_ca, label = "calm window", color = :steelblue)
savefig(p_roll, "fig_rolling_gap.png")
p_roll

println("\n#### CELL 14")
se = vec(std(Rest, dims=1)) ./ sqrt(T)   # SE_i of industry i's mean; the d above is 2 se

p_k = plot(xlabel = "Γ (industries the adversary may affect)", legend = :topright,
           ylabel = "Worst-case annualized return (%)", size = (760, 460),
           title = "Price of robustness under three uncertainty scales")
for k in (1, 2, 3)
    arm = [solve_robust(μ, k .* se, Γ) for Γ in Γ_range]   # the same sweep, one d per k
    Wk = [ann(a[2]) for a in arm]
    Hk = [count(a[1] .> 1e-4) for a in arm]                # holdings, above solver dust
    # the collapse budget is the start of the terminal run of single-industry books, not
    # the first budget holding one, which would be Γ = 0 on every arm.
    @printf("k = %d: peak %2d industries, collapses at Γ = %2d to %-6s plateau %7.2f%%\n",
            k, maximum(Hk), Γ_range[findlast(Hk .> 1) + 1],
            industries[argmax(arm[end][1])], Wk[end])
    plot!(p_k, Γ_range, Wk, label = "k = $k")
end
savefig(p_k, "fig_k_curves.png")
p_k

