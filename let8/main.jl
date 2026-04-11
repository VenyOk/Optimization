using Plots
using Printf

f1(x) = 5 - 24x[1] + 17x[1]^2 - (11/3)*x[1]^3 + (1/4)*x[1]^4
f2(x) = x[1]^2 * (x[1] - 2)^2 * (x[1] + 2)^2 / 8 + 0.3*x[1]

criteria = [f1, f2]
labels   = ["f₁", "f₂"]
colors   = [:cadetblue, :darkorchid]
linestyles = [:solid, :dash]

function search(
    criteria,
    x0::Vector{Float64},
    weights::Vector{Float64};
    T0::Float64    = 5.0,
    alpha::Float64 = 0.92,
    step::Float64  = 1.0,
    max_iter::Int  = 20,
    sa_steps::Int  = 60,
)
    x           = copy(x0)
    n           = length(x)
    T           = T0
    history     = [copy(x)]
    stop_reason = "лимит итераций"
    w           = weights ./ sum(weights)

    function approx_min(f, n_steps=400)
        best_v = f(x)
        for _ in 1:n_steps
            x_try = x .+ (2 .* rand(n) .- 1) .* step * 3
            v = f(x_try)
            if v < best_v; best_v = v end
        end
        return best_v
    end

    for iter in 1:max_iter
        ideal_vals = [approx_min(f) for f in criteria]
        dist_fn(xt) = sqrt(sum(w[i] * (criteria[i](xt) - ideal_vals[i])^2 for i in eachindex(criteria)))

        cur_d  = dist_fn(x)
        best_x = copy(x)
        best_d = cur_d

        for _ in 1:sa_steps
            x_try = x .+ (2 .* rand(n) .- 1) .* step
            d_try = dist_fn(x_try)
            dD = d_try - cur_d
            if dD < 0 || rand() < exp(-dD / T)
                x = x_try; cur_d = d_try
            end
            if cur_d < best_d
                best_d = cur_d; best_x = copy(x)
            end
        end

        x = best_x
        push!(history, copy(x))
        T *= alpha
    end

    return x, history, stop_reason
end

function plot_all(weights, gif_name, variant)
    default(
        framestyle = :box,
        grid = false,
        minorgrid = false,
        titlefontsize = 11,
        guidefontsize = 10,
        legendfontsize = 8,
        background_color_subplot = :white,
        background_color_inside = :white,
    )

    x0 = [5.0]
    x_opt, history, stop_rs = search(criteria, x0, weights)

    xs      = range(-5.0, 10.0, length=900)
    fvals   = [[f([xi]) for xi in xs] for f in criteria]
    ideal_v = [minimum(f([xi]) for xi in xs) for f in criteria]
    w       = weights ./ sum(weights)
    Dvals   = [sqrt(sum(w[i] * (criteria[i]([xi]) - ideal_v[i])^2 for i in eachindex(criteria))) for xi in xs]

    ylim_f = (-10.0, 20.0)

    D_start = sqrt(sum(w[i]*(criteria[i](history[1])-ideal_v[i])^2 for i in eachindex(criteria)))
    D_opt   = sqrt(sum(w[i]*(criteria[i](x_opt)   -ideal_v[i])^2 for i in eachindex(criteria)))
    p_d = plot(
        xs,
        Dvals;
        label = "D",
        color = :sienna,
        lw = 2.3,
        linealpha = 0.92,
        fillrange = 0,
        fillalpha = 0.14,
        fillcolor = :sienna,
        title = "Расстояние",
        xlabel = "x",
        ylabel = "D",
        legend = :topright,
        size = (780, 390),
    )
    scatter!(p_d, [history[1][1]], [D_start]; color = :dimgray, ms = 7, marker = :circle, label = "старт")
    scatter!(p_d, [x_opt[1]], [D_opt]; color = :firebrick, ms = 9, marker = :star5, label = "итог")
    display(p_d)

    p_f = plot(
        title = @sprintf("Критерии, %s, x*=%.4f", variant, x_opt[1]),
        xlabel = "x",
        ylabel = "f",
        legend = :outertopright,
        size = (800, 430),
        ylims = ylim_f,
    )
    for k in eachindex(criteria)
        plot!(
            p_f,
            xs,
            fvals[k];
            label = labels[k],
            color = colors[k],
            alpha = 0.82,
            lw = k == 1 ? 2.4 : 2.0,
            ls = linestyles[k],
        )
    end
    for k in eachindex(criteria)
        scatter!(p_f, [history[1][1]], [criteria[k](history[1])];
                 color = :dimgray, ms = 7, marker = :circle, label = k == 1 ? "старт" : "")
    end
    for k in eachindex(criteria)
        scatter!(p_f, [x_opt[1]], [criteria[k](x_opt)];
                 color = :firebrick, ms = 8, marker = :star5, label = k == 1 ? "итог" : "")
    end
    display(p_f)

    anim = @animate for (i, x_cur) in enumerate(history)
        is_last = (i == length(history))
        title_str = is_last ?
            @sprintf("%s, шаг %d, x=%.3f (%s)", variant, i - 1, x_cur[1], stop_rs) :
            @sprintf("%s, шаг %d, x=%.3f", variant, i - 1, x_cur[1])
        p = plot(
            title = title_str,
            xlabel = "x",
            ylabel = "f",
            legend = :outertopright,
            size = (800, 430),
            ylims = ylim_f,
        )
        for k in eachindex(criteria)
            plot!(
                p,
                xs,
                fvals[k];
                label = labels[k],
                color = colors[k],
                alpha = 0.82,
                lw = k == 1 ? 2.4 : 2.0,
                ls = linestyles[k],
            )
        end
        for k in eachindex(criteria)
            scatter!(p, [history[1][1]], [criteria[k](history[1])];
                     color = :dimgray, ms = 6, marker = :circle, label = k == 1 ? "старт" : "")
        end
        for k in eachindex(criteria)
            scatter!(p, [x_cur[1]], [criteria[k](x_cur)]; color = colors[k], ms = 8, label = "")
        end
        p
    end
    gif(anim, gif_name; fps=2, show_msg=false)

    xs_p  = collect(range(-5.0, 10.0, length=2000))
    F     = hcat([[f([xi]) for f in criteria] for xi in xs_p]...)
    dominated = [any(j -> j != i && all(F[:,j] .<= F[:,i]) && any(F[:,j] .< F[:,i]), axes(F,2))
                 for i in axes(F,2)]
    F_par = F[:, .!dominated]
    F_opt = [f(x_opt) for f in criteria]

    p_par = plot(
        title = "Парето",
        xlabel = labels[1],
        ylabel = labels[2],
        legend = :topright,
        size = (780, 420),
    )
    ord = sortperm(F_par[1, :])
    plot!(
        p_par,
        F_par[1, ord],
        F_par[2, ord];
        color = :tan,
        lw = 1.4,
        linealpha = 0.55,
        label = "",
    )
    scatter!(p_par, F_par[1, :], F_par[2, :]; color = :darkcyan, ms = 3.5, alpha = 0.75, label = "фронт")
    scatter!(p_par, [F_opt[1]], [F_opt[2]]; color = :firebrick, ms = 10, marker = :star5, label = "итог")
    display(p_par)

    println(@sprintf("%s: x*=%.5f, f1=%.4f, f2=%.4f, шагов=%d",
            variant, x_opt[1], f1(x_opt), f2(x_opt), length(history) - 1))
end

weights = [1.0, 1.0]
plot_all(weights, "convergence_equal.gif", "равные веса")

weights = [67.0, 1.0]
plot_all(weights, "convergence_f1.gif", "веса 67 и 1")

weights = [1.0, 67.0]
plot_all(weights, "convergence_f2.gif", "веса 1 и 67")
