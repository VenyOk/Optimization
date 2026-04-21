using Random
using Printf
using Plots

function f9(x::AbstractVector{<:Real})
    n = 3
    s = 0.0
    for i in 1:4
        ci = 2.0 * i
        t = 0.0
        for j in 1:n
            aij = j / ci
            pij = aij^2
            d = x[j] - pij
            t += aij * d * d
        end
        s += ci * exp(-t)
    end
    -s
end

rand_cell(rng, a, b) = a .+ (b .- a) .* rand(rng, length(a))

function mutate_clone(rng, x, a, b, r)
    n = length(x)
    y = Vector{Float64}(undef, n)
    for i in 1:n
        while true
            u = rand(rng)
            if u > 0.5
                y[i] = x[i] + rand(rng) * (b[i] - x[i]) * r
            else
                y[i] = x[i] - rand(rng) * (x[i] - a[i]) * r
            end
            (a[i] <= y[i] <= b[i]) && break
        end
    end
    y
end

function immune_optimize(f, a, b; Np=50, s=10, d=10, K=60, clone=:proportional, η=0.6, Nc=10, r=0.2, seed=42, save_history=false)
    rng = MersenneTwister(seed)
    pop = [rand_cell(rng, a, b) for _ in 1:Np]
    fit = [f(x) for x in pop]
    bestx = pop[argmin(fit)]
    bestf = minimum(fit)
    hist_bestf = save_history ? Vector{Float64}(undef, K + 1) : Float64[]
    hist_bestx = save_history ? Matrix{Float64}(undef, length(a), K + 1) : zeros(0, 0)
    hist_pop = save_history ? Vector{Matrix{Float64}}(undef, K + 1) : Matrix{Float64}[]
    if save_history
        hist_bestf[1] = bestf
        hist_bestx[:, 1] = bestx
        hist_pop[1] = reduce(hcat, pop)
    end

    for k in 1:K
        ord = sortperm(fit)
        pop = pop[ord]
        fit = fit[ord]

        for j in 1:min(s, Np)
            xpar = pop[j]
            fpar = fit[j]
            Nj = clone === :uniform ? Nc : max(1, Int(floor(η * Np / j)))
            best_y = xpar
            best_fy = fpar
            for _ in 1:Nj
                y = mutate_clone(rng, xpar, a, b, r)
                fy = f(y)
                if fy < best_fy
                    best_fy = fy
                    best_y = y
                end
            end
            if best_fy < fpar
                pop[j] = best_y
                fit[j] = best_fy
            end
        end

        ord2 = sortperm(fit)
        pop = pop[ord2]
        fit = fit[ord2]

        dd = min(d, Np)
        for idx in (Np - dd + 1):Np
            pop[idx] = rand_cell(rng, a, b)
            fit[idx] = f(pop[idx])
        end

        kbest = argmin(fit)
        if fit[kbest] < bestf
            bestf = fit[kbest]
            bestx = pop[kbest]
        end
        if save_history
            hist_bestf[k + 1] = bestf
            hist_bestx[:, k + 1] = bestx
            hist_pop[k + 1] = reduce(hcat, pop)
        end
    end

    if save_history
        return bestx, bestf, hist_bestf, hist_bestx, hist_pop
    end
    return bestx, bestf
end

function save_visualization(outdir, a, b, hist_bestf, hist_bestx, hist_pop; fps=5)
    mkpath(outdir)
    it = 0:(length(hist_bestf) - 1)
    p1 = plot(it, hist_bestf; xlabel="k", ylabel="f(best)", legend=false, lw=2, color=:firebrick, title="Сходимость")
    savefig(p1, joinpath(outdir, "convergence.png"))

    p2 = plot(it, hist_bestx[1, :]; xlabel="k", ylabel="x", label="x1", lw=2)
    plot!(p2, it, hist_bestx[2, :]; label="x2", lw=2)
    plot!(p2, it, hist_bestx[3, :]; label="x3", lw=2)
    savefig(p2, joinpath(outdir, "best_x.png"))

    p3 = plot(hist_bestx[1, :], hist_bestx[2, :], hist_bestx[3, :]; seriestype=:path, lw=2, color=:navy, legend=false, xlabel="x1", ylabel="x2", zlabel="x3", title="Траектория лучшего x")
    scatter!(p3, [hist_bestx[1, 1]], [hist_bestx[2, 1]], [hist_bestx[3, 1]]; ms=6, color=:gray)
    scatter!(p3, [hist_bestx[1, end]], [hist_bestx[2, end]], [hist_bestx[3, end]]; ms=7, color=:firebrick, marker=:star5)
    savefig(p3, joinpath(outdir, "best_path_3d.png"))

    anim = @animate for k in 1:length(hist_pop)
        P = hist_pop[k]
        xs = P[1, :]
        ys = P[2, :]
        zs = P[3, :]
        bx, by, bz = hist_bestx[1, k], hist_bestx[2, k], hist_bestx[3, k]
        plt = plot(xlims=(a[1], b[1]), ylims=(a[2], b[2]), zlims=(a[3], b[3]), xlabel="x1", ylabel="x2", zlabel="x3", title="k=$(k-1)")
        scatter!(plt, xs, ys, zs; ms=3.5, alpha=0.35, color=:darkcyan, label=false)
        scatter!(plt, [bx], [by], [bz]; ms=7, color=:firebrick, marker=:star5, label=false)
        plt
    end
    gif(anim, joinpath(outdir, "search.gif"); fps=fps, show_msg=false)
end

function main()
    a = fill(2.0, 3)
    b = fill(13.0, 3)
    x, fx, hist_bestf, hist_bestx, hist_pop = immune_optimize(f9, a, b; Np=50, s=10, d=10, K=80, clone=:proportional, η=0.6, r=0.2, seed=67, save_history=true)
    save_visualization(joinpath(@__DIR__, "output"), a, b, hist_bestf, hist_bestx, hist_pop; fps=5)
    @printf("x* = [%.6f, %.6f, %.6f]\n", x[1], x[2], x[3])
    @printf("f(x*) = %.10f\n", fx)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
