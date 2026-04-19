using Plots
using Printf
using Random

f1(x) = 5 - 24x[1] + 17x[1]^2 - (11 / 3) * x[1]^3 + (1 / 4) * x[1]^4
f2(x) = (x[1]^2 - 2.3 * x[1] + 0.6)^2 / 8 + 0.1 * (x[1] - 0.3)^2 + 0.12 * x[1]

criteria = [f1, f2]
labels = ["f1", "f2"]
colors = [:cadetblue, :darkorchid]
linestyles = [:solid, :dash]

function war_ranks(F, v)
    S, m = size(F)
    r = zeros(S)
    for k in 1:m
        ord = sortperm(F[:, k])
        for pos in 1:S
            i = ord[pos]
            r[i] += v[k] * pos
        end
    end
    r
end

function war_score_at(xt, x_grid, criteria, w)
    n = length(x_grid)
    m = length(criteria)
    F = zeros(n + 1, m)
    for i in 1:n
        xi = [x_grid[i]]
        for k in 1:m
            F[i, k] = criteria[k](xi)
        end
    end
    for k in 1:m
        F[n + 1, k] = criteria[k](xt)
    end
    r = war_ranks(F, w)
    r[end]
end

function search_war(
    criteria,
    x0::Vector{Float64},
    weights::Vector{Float64},
    x_grid::Vector{Float64};
    T0::Float64 = 5.0,
    alpha::Float64 = 0.92,
    step::Float64 = 1.0,
    max_iter::Int = 20,
    sa_steps::Int = 60,
)
    x = copy(x0)
    n = length(x)
    T = T0
    history = [copy(x)]
    stop_reason = "лимит итераций"
    w = weights ./ sum(weights)
    lo = x_grid[1]
    hi = x_grid[end]
    x .= clamp.(x, lo, hi)

    for _ in 1:max_iter
        cur_w = war_score_at(x, x_grid, criteria, w)
        best_x = copy(x)
        best_w = cur_w

        for _ in 1:sa_steps
            x_try = clamp.(x .+ (2 .* rand(n) .- 1) .* step, lo, hi)
            w_try = war_score_at(x_try, x_grid, criteria, w)
            dw = w_try - cur_w
            if dw < 0 || rand() < exp(-dw / T)
                x = x_try
                cur_w = w_try
            end
            if cur_w < best_w
                best_w = cur_w
                best_x = copy(x)
            end
        end

        x = best_x
        push!(history, copy(x))
        T *= alpha
    end

    return x, history, stop_reason
end

function plot_all_war(weights, gif_name, variant)
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

    x_grid = collect(range(-5.0, 10.0, length = 500))
    w = weights ./ sum(weights)

    x0 = [5.0]
    x_opt, history, stop_rs = search_war(criteria, x0, weights, x_grid)

    xs = range(-5.0, 10.0, length = 900)
    fvals = [[f([xi]) for xi in xs] for f in criteria]
    Wvals = [war_score_at([xi], x_grid, criteria, w) for xi in xs]

    ylim_f = (-10.0, 20.0)

    W_start = war_score_at(history[1], x_grid, criteria, w)
    W_opt = war_score_at(x_opt, x_grid, criteria, w)

    p_w = plot(
        xs,
        Wvals;
        label = "WAR",
        color = :sienna,
        lw = 2.3,
        linealpha = 0.92,
        fillrange = 0,
        fillalpha = 0.14,
        fillcolor = :sienna,
        title = "",
        xlabel = "x",
        ylabel = "WAR",
        legend = :topright,
        size = (780, 390),
    )
    scatter!(p_w, [history[1][1]], [W_start]; color = :dimgray, ms = 7, marker = :circle, label = "старт")
    scatter!(p_w, [x_opt[1]], [W_opt]; color = :firebrick, ms = 9, marker = :star5, label = "итог")

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
        scatter!(
            p_f,
            [history[1][1]],
            [criteria[k](history[1])];
            color = :dimgray,
            ms = 7,
            marker = :circle,
            label = k == 1 ? "старт" : "",
        )
    end
    for k in eachindex(criteria)
        scatter!(
            p_f,
            [x_opt[1]],
            [criteria[k](x_opt)];
            color = :firebrick,
            ms = 8,
            marker = :star5,
            label = k == 1 ? "итог" : "",
        )
    end

    anim = @animate for (i, x_cur) in enumerate(history)
        is_last = (i == length(history))
        title_str =
            is_last ? @sprintf("%s, шаг %d, x=%.3f (%s)", variant, i - 1, x_cur[1], stop_rs) :
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
            scatter!(
                p,
                [history[1][1]],
                [criteria[k](history[1])];
                color = :dimgray,
                ms = 6,
                marker = :circle,
                label = k == 1 ? "старт" : "",
            )
        end
        for k in eachindex(criteria)
            scatter!(p, [x_cur[1]], [criteria[k](x_cur)]; color = colors[k], ms = 8, label = "")
        end
        p
    end
    gif(anim, joinpath(@__DIR__, gif_name); fps = 2, show_msg = false)

    xs_p = collect(range(-5.0, 10.0, length = 2000))
    F = hcat([[f([xi]) for f in criteria] for xi in xs_p]...)
    dominated = [
        any(
            j -> j != i && all(F[:, j] .<= F[:, i]) && any(F[:, j] .< F[:, i]),
            axes(F, 2),
        ) for i in axes(F, 2)
    ]
    F_par = F[:, .!dominated]
    F_opt = [f(x_opt) for f in criteria]

    p_par = plot(
        title = "Фронт Парето",
        xlabel = labels[1],
        ylabel = labels[2],
        legend = :topright,
        size = (780, 420),
    )
    scatter!(
        p_par,
        F_par[1, :],
        F_par[2, :];
        color = :darkcyan,
        ms = 3.5,
        alpha = 0.75,
        label = "фронт Парето",
    )
    scatter!(p_par, [F_opt[1]], [F_opt[2]]; color = :firebrick, ms = 10, marker = :star5, label = "итог WAR")

    png_par = joinpath(@__DIR__, replace(gif_name, ".gif" => "_pareto.png"))
    savefig(p_par, png_par)
    println(
        @sprintf(
            "%s: x*=%.5f, f1=%.4f, f2=%.4f, шагов=%d",
            variant,
            x_opt[1],
            f1(x_opt),
            f2(x_opt),
            length(history) - 1,
        ),
    )
end

function main()
    Random.seed!(67)
    plot_all_war([1.0, 1.0], "war_equal.gif", "равные веса")
    plot_all_war([4.0, 1.0], "war_f1.gif", "веса 4 и 1")
    plot_all_war([1.0, 4.0], "war_f2.gif", "веса 1 и 4")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
