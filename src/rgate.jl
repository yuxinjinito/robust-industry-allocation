# R-gate: reprint every baseline output through the harness and compare it, cell
# by cell and line by line, with baseline/saved.txt. The print statements are the baseline's
# (nb.jl cells 1-14, plots removed); every number comes from harness functions, and the
# holdout and rolling books come from run_arms, the code path the extension arms use.

function rgate_print(io, dates, industries, R)
    cell(k) = println(io, "\n#### CELL $k")
    cell(1)
    sel_all = win(dates, 20150101, 20251231)
    @assert length(industries) == 30
    @assert sum(sel_all) == 2766
    @assert all(R[sel_all, :] .> -0.98)
    println(io, "data checks passed")

    cell(2)
    sel_est = win(dates, 20150101, 20221231)
    Rest = R[sel_est, :]
    T = size(Rest, 1)
    μ, d = fit_inputs(Rest)
    println(io, "annualized mean returns: min ", round(minimum(ann.(μ)), digits=1),
            "%, max ", round(maximum(ann.(μ)), digits=1), "%")
    println(io, "annualized 2xSE:         min ", round(minimum(ann.(d)), digits=1),
            "%, max ", round(maximum(ann.(d)), digits=1), "%")
    C = cor(Rest)
    println(io, "average pairwise correlation: ", round((sum(C) - 30) / (30 * 29), digits=2))

    cell(3)
    cell(4)
    worst_direct(x, μ, d, Γ) = sum(μ .* x) - sum(sort(d .* x, rev=true)[1:Γ])
    function subsets_of_size(n, k)
        k == 0 && return [Int[]]
        result = Vector{Vector{Int}}()
        for last in k:n, head in subsets_of_size(last - 1, k - 1)
            push!(result, vcat(head, last))
        end
        return result
    end
    function solve_enum(μ, d, Γ)
        n = length(μ)
        m = Model(HiGHS.Optimizer); set_silent(m)
        @variable(m, x[1:n] >= 0); @variable(m, t)
        @constraint(m, sum(x) == 1)
        for S in subsets_of_size(n, Γ)
            @constraint(m, t <= sum(μ[i] * x[i] for i in 1:n) - sum(d[i] * x[i] for i in S; init = 0.0))
        end
        @objective(m, Max, t); optimize!(m)
        check_status(m, :highs, "enum Γ=$Γ")
        return value.(x), objective_value(m)
    end
    ntoy = 5
    μ_toy, d_toy = μ[1:ntoy], d[1:ntoy]
    println(io, "toy industries: ", join(industries[1:ntoy], ", "))
    for Γ in 0:ntoy
        x1, o1, _ = solve_robust(μ_toy, d_toy, Γ)
        x2, o2 = solve_enum(μ_toy, d_toy, Γ)
        w1 = worst_direct(x1, μ_toy, d_toy, Γ)
        diff = max(abs(o1 - o2), abs(o1 - w1))
        println(io, "Γ = ", Γ, ": dual LP = ", round(o1, sigdigits=6),
                ", enumeration LP = ", round(o2, sigdigits=6),
                ", direct sort = ", round(w1, sigdigits=6),
                ", max diff = ", round(diff, sigdigits=3))
        @assert diff < 1e-9
    end
    println(io, "toy validation passed")

    cell(5)
    println(io, "full model on all 30 industries:")
    for Γ in (0, 5, 15, 30)
        x, obj, z = solve_robust(μ, d, Γ)
        holdings = [(industries[i], 100 * x[i]) for i in 1:30 if x[i] > 1e-4]
        sort!(holdings, by = h -> -h[2])
        top5 = join([h[1] * " " * string(round(h[2], digits=1)) * "%" for h in holdings[1:min(5, end)]], ", ")
        println(io, "Γ = ", Γ, ": worst-case return ", round(ann(obj), digits=2), "%, ",
                length(holdings), " holdings; largest: ", top5)
    end

    cell(6)
    Γ_range = 0:30
    sols = [solve_robust(μ, d, Γ) for Γ in Γ_range]
    worst_case = [s[2] for s in sols]; z_values = [s[3] for s in sols]
    println(io, "slope of the curve vs -z* (equal on linear stretches):")
    for k in 1:10
        println(io, "Γ = ", Γ_range[k], ": slope = ", round(worst_case[k+1] - worst_case[k], sigdigits=4),
                ", -z* = ", round(-z_values[k], sigdigits=4))
    end
    cell(7); cell(8)

    cell(9)
    hm = holdout_masks(dates)
    @assert (hm.tr .| hm.va) == sel_est
    ftr, fref = window_fits(dates, R, hm)
    Gsel, X_tr, sh_va = select_h(build_box, GRID_G, ftr, R[hm.va, :])
    books_ho = run_arms(ftr, fref, R[hm.va, :]; arms = [:NOM, :EW, :BOX], ctx = "holdout")
    @assert books_ho[:BOX].h == Gsel
    Gamma_star = Gsel
    println(io, "validation Sharpe by budget, fitted on 2015-2020 and scored on 2021-2022")
    for G in (0, 1, 11, 13)
        @printf(io, "  Γ = %2d: %2d holdings, Sharpe %7.4f\n", G, count(X_tr[G+1] .> held_thresh), sh_va[G+1])
    end
    @printf(io, "selected Γ* = %d\n", Gamma_star)

    cell(10)
    x_rob, x_nom, x_ew = books_ho[:BOX].x, books_ho[:NOM].x, books_ho[:EW].x
    books = [("robust Γ=$Gamma_star", x_rob), ("nominal Γ=0", x_nom), ("equal weight 1/N", x_ew)]
    @printf(io, "LP worst-case bound %.2f%% at Γ* = %d, against %.2f%% at Γ = 0\n",
            ann(solve_robust(μ, d, Gamma_star)[2]), Gamma_star, ann(solve_robust(μ, d, 0)[2]))
    @printf(io, "  the second is just the annualized in-sample mean of %s, the best single industry\n",
            industries[argmax(μ)])
    Rho = R[hm.te, :]
    println(io, "holdout 2023-2025     hold  ann_ret%  ann_vol%   sharpe   maxDD%")
    for (lab, x) in books
        r = Rho * x
        @printf(io, "  %-18s %3d %9.2f %9.2f %8.4f %8.2f\n", lab, count(x .> held_thresh),
                ann_ret(r), ann_vol(r), sharpe(r), max_drawdown(r))
    end

    cell(11)
    mu_ho = vec(mean(Rho, dims=1))
    below = mu_ho .< (μ .- d)
    @printf(io, "industries whose 2023-2025 mean fell below μ̂ - d: %d of 30 (%s)\n",
            count(below), join(industries[below], ", "))
    @printf(io, "%s, the whole nominal book, ranked %d of 30 on holdout mean\n",
            industries[argmax(μ)], findfirst(==(argmax(μ)), sortperm(mu_ho, rev=true)))
    keep = dates .<= 20221231
    dk, Rk = dates[keep], R[keep, :]
    hk = holdout_masks(dk)
    ftr_c, fref_c = window_fits(dk, Rk, hk)
    bc = run_arms(ftr_c, fref_c, Rk[hk.va, :]; arms = [:BOX], ctx = "holdout truncated")
    @printf(io, "leakage audit: post-2022 rows deleted, Γ* = %d, largest weight difference %g\n",
            bc[:BOX].h, maximum(abs.(bc[:BOX].x .- x_rob)))
    cell(12)

    cell(13)
    rounds = []
    for w in rolling_windows(dates)
        ft, fr = window_fits(dates, R, w)
        b = run_arms(ft, fr, R[w.va, :]; arms = [:NOM, :EW, :BOX], ctx = "rolling $(w.y0)")
        Rte = R[w.te, :]
        push!(rounds, (y0 = w.y0, G = b[:BOX].h,
                       ret_rob = ann_ret(Rte * b[:BOX].x), ret_nom = ann_ret(Rte * b[:NOM].x),
                       ret_ew = ann_ret(Rte * b[:EW].x),
                       dd_rob = max_drawdown(Rte * b[:BOX].x), dd_nom = max_drawdown(Rte * b[:NOM].x),
                       dd_ew = max_drawdown(Rte * b[:EW].x)))
    end
    paid = [r for r in rounds if r.G > 0]
    @printf(io, "%d test windows, %d-%d. Γ* = 0 in %d of them, so the model bought protection in %d\n",
            length(rounds), rounds[1].y0, rounds[end].y0 + 2,
            length(rounds) - length(paid), length(paid))
    gap_dd = [r.dd_rob - r.dd_nom for r in paid]
    gap_ret = [r.ret_rob - r.ret_nom for r in paid]
    println(io, "\nrobust minus nominal, so a negative drawdown gap and a positive return gap favour robust")
    println(io, "                     robust better     median   middle half")
    @printf(io, "  maximum drawdown %10d of %-3d %8.1f pp %6.1f to %5.1f\n", count(<(0), gap_dd),
            length(paid), median(gap_dd), quantile(gap_dd, .25), quantile(gap_dd, .75))
    @printf(io, "  annual return    %10d of %-3d %8.1f pp %6.1f to %5.1f\n", count(>(0), gap_ret),
            length(paid), median(gap_ret), quantile(gap_ret, .25), quantile(gap_ret, .75))
    st = [r for r in paid if r.dd_ew > 20]
    ca = [r for r in paid if r.dd_ew <= 20]
    @printf(io, "\ndrawdown gap by market condition: %.1f pp over the %d stressed rounds, %.1f pp over the %d calm ones\n",
            median([r.dd_rob - r.dd_nom for r in st]), length(st),
            median([r.dd_rob - r.dd_nom for r in ca]), length(ca))
    gap_ew = [r.dd_rob - r.dd_ew for r in paid]
    @printf(io, "against equal weight instead: smaller drawdown in %d of %d rounds, median gap %.1f pp\n",
            count(<(0), gap_ew), length(paid), median(gap_ew))

    cell(14)
    se = d ./ 2
    for k in (1, 2, 3)
        arm = [solve_robust(μ, k .* se, Γ) for Γ in Γ_range]
        Wk = [ann(a[2]) for a in arm]
        Hk = [count(a[1] .> 1e-4) for a in arm]
        @printf(io, "k = %d: peak %2d industries, collapses at Γ = %2d to %-6s plateau %7.2f%%\n",
                k, maximum(Hk), Γ_range[findlast(Hk .> 1) + 1],
                industries[argmax(arm[end][1])], Wk[end])
    end
end

# split "#### CELL k" blocks into k => non-empty lines, trailing spaces stripped
function cells_of(text)
    out = Dict{Int,Vector{String}}(); k = 0
    for l in split(text, '\n')
        m = match(r"^#### CELL (\d+)$", l)
        if m !== nothing
            k = parse(Int, m.captures[1]); out[k] = String[]
        elseif k > 0 && !isempty(strip(l))
            push!(out[k], rstrip(l))
        end
    end
    return out
end

function rgate(dates, industries, R, saved_path)
    io = IOBuffer()
    rgate_print(io, dates, industries, R)
    got = cells_of(String(take!(io)))
    want = cells_of(read(saved_path, String))
    bad = String[]
    for k in sort(collect(keys(want)))
        g = get(got, k, String[])
        g == want[k] && continue
        push!(bad, "cell $k:\n  want: " * join(want[k], "\n        ") * "\n  got:  " * join(g, "\n        "))
    end
    nlines = sum(length(v) for v in values(want))
    return isempty(bad), nlines, bad
end
