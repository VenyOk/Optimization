using Random
using Printf
using Plots
using Logging

global_logger(ConsoleLogger(stderr, Logging.Warn))

poly4(x) = 5 - 24x + 17x^2 - (11 / 3) * x^3 + (1 / 4) * x^4
phi_method(x) = (x^2 - 4)^2 + 0.5 * x
rastrigin2d(v::NTuple{2, Float64}) = 20 + v[1]^2 + v[2]^2 - 10 * (cos(2 * pi * v[1]) + cos(2 * pi * v[2]))

struct SAProblem
    name::String
    f::Function
    a::Float64
    b::Float64
    h::Float64
    x0
    neighborhood::Symbol
    dim::Int
end

struct SAConfig
    max_iters::Int
    eps::Float64
    T0::Float64
    alpha::Float64
    beta::Float64
    dim::Int
    seed::Int
end

function make_grid(a::Float64, b::Float64, h::Float64)
    n = Int(round((b - a) / h))
    a .+ (0:n) .* h
end

function nearest_index(x::Float64, a::Float64, h::Float64, n::Int)
    idx = Int(round((x - a) / h)) + 1
    clamp(idx, 1, n)
end

function reflect_index(i::Int, step::Int, n::Int)
    j = i + step
    while j < 1 || j > n
        if j < 1
            j = 2 - j
        elseif j > n
            j = 2 * n - j
        end
    end
    j
end

temp_geometric(t::Int, cfg::SAConfig) = cfg.T0 * cfg.alpha^(t - 1)
temp_logarithmic(t::Int, cfg::SAConfig) = cfg.T0 / log(t + 1)
temp_very_fast(t::Int, cfg::SAConfig, dim::Int) = cfg.T0 * exp(-cfg.beta * t^(1 / dim))

function temperature(schedule::Symbol, t::Int, cfg::SAConfig, dim::Int)
    if schedule == :geometric
        return temp_geometric(t, cfg)
    elseif schedule == :logarithmic
        return temp_logarithmic(t, cfg)
    elseif schedule == :very_fast
        return temp_very_fast(t, cfg, dim)
    end
    error("Unknown cooling schedule: $schedule")
end

function grid_optimum(problem::SAProblem)
    xs = make_grid(problem.a, problem.b, problem.h)
    if problem.dim == 1
        ys = problem.f.(xs)
        k = argmin(ys)
        return xs[k], ys[k]
    end

    best_x = (xs[1], xs[1])
    best_f = Inf
    for x1 in xs
        for x2 in xs
            y = problem.f((x1, x2))
            if y < best_f
                best_f = y
                best_x = (x1, x2)
            end
        end
    end
    best_x, best_f
end

function run_sa(problem::SAProblem, schedule::Symbol, cfg::SAConfig)
    rng = MersenneTwister(cfg.seed)
    xs = make_grid(problem.a, problem.b, problem.h)
    n = length(xs)

    x_opt, f_opt = grid_optimum(problem)
    hit_iter = -1
    hit_time_s = -1.0
    t_start = time_ns()

    if problem.dim == 1
        i = nearest_index(problem.x0, problem.a, problem.h, n)
        x = xs[i]
        y = problem.f(x)

        best_x = x
        best_f = y
        x_hist = Float64[x]
        f_hist = Float64[y]
        temp_hist = Float64[]

        for t in 1:cfg.max_iters
            T = max(temperature(schedule, t, cfg, problem.dim), 1e-12)
            push!(temp_hist, T)

            step = if problem.neighborhood == :unit
                rand(rng, Bool) ? 1 : -1
            else
                max_jump = max(1, Int(round(0.5 * (n - 1))))
                jump = Int(round(max_jump * T / cfg.T0))
                jump = clamp(jump, 1, max_jump)
                s = rand(rng, -jump:jump)
                while s == 0
                    s = rand(rng, -jump:jump)
                end
                s
            end

            j = reflect_index(i, step, n)
            x_new = xs[j]
            y_new = problem.f(x_new)
            d = y_new - y

            if d <= 0 || rand(rng) < exp(-d / T)
                i = j
                x = x_new
                y = y_new
            end

            if y < best_f
                best_f = y
                best_x = x
            end

            push!(x_hist, x)
            push!(f_hist, y)

            if hit_iter == -1 && abs(best_f - f_opt) <= cfg.eps
                hit_iter = t
                hit_time_s = (time_ns() - t_start) / 1e9
            end
        end

        total_time_s = (time_ns() - t_start) / 1e9
        if hit_iter == -1
            hit_time_s = total_time_s
        end

        return (
            schedule = schedule,
            x_hist = x_hist,
            f_hist = f_hist,
            temp_hist = temp_hist,
            best_x = best_x,
            best_f = best_f,
            x_opt = x_opt,
            f_opt = f_opt,
            hit_iter = hit_iter,
            hit_time_s = hit_time_s,
            total_time_s = total_time_s
        )
    end

    i1 = nearest_index(problem.x0[1], problem.a, problem.h, n)
    i2 = nearest_index(problem.x0[2], problem.a, problem.h, n)
    x = (xs[i1], xs[i2])
    y = problem.f(x)

    best_x = x
    best_f = y
    x_hist = [(x[1], x[2])]
    f_hist = Float64[y]
    temp_hist = Float64[]

    for t in 1:cfg.max_iters
        T = max(temperature(schedule, t, cfg, problem.dim), 1e-12)
        push!(temp_hist, T)

        max_jump = max(1, Int(round(0.5 * (n - 1))))
        jump = Int(round(max_jump * T / cfg.T0))
        jump = clamp(jump, 1, max_jump)

        s1 = rand(rng, -jump:jump)
        s2 = rand(rng, -jump:jump)
        while s1 == 0 && s2 == 0
            s1 = rand(rng, -jump:jump)
            s2 = rand(rng, -jump:jump)
        end

        j1 = reflect_index(i1, s1, n)
        j2 = reflect_index(i2, s2, n)
        x_new = (xs[j1], xs[j2])
        y_new = problem.f(x_new)
        d = y_new - y

        if d <= 0 || rand(rng) < exp(-d / T)
            i1 = j1
            i2 = j2
            x = x_new
            y = y_new
        end

        if y < best_f
            best_f = y
            best_x = x
        end

        push!(x_hist, (x[1], x[2]))
        push!(f_hist, y)

        if hit_iter == -1 && abs(best_f - f_opt) <= cfg.eps
            hit_iter = t
            hit_time_s = (time_ns() - t_start) / 1e9
        end
    end

    total_time_s = (time_ns() - t_start) / 1e9
    if hit_iter == -1
        hit_time_s = total_time_s
    end

    (
        schedule = schedule,
        x_hist = x_hist,
        f_hist = f_hist,
        temp_hist = temp_hist,
        best_x = best_x,
        best_f = best_f,
        x_opt = x_opt,
        f_opt = f_opt,
        hit_iter = hit_iter,
        hit_time_s = hit_time_s,
        total_time_s = total_time_s
    )
end

function schedule_label(schedule::Symbol)
    if schedule == :geometric
        return "Geometric"
    elseif schedule == :logarithmic
        return "Logarithmic"
    elseif schedule == :very_fast
        return "Very fast"
    end
    string(schedule)
end

function schedule_slug(schedule::Symbol)
    if schedule == :geometric
        return "geometric"
    elseif schedule == :logarithmic
        return "logarithmic"
    elseif schedule == :very_fast
        return "very_fast"
    end
    string(schedule)
end

function schedule_color(schedule::Symbol)
    if schedule == :geometric
        return :blue
    elseif schedule == :logarithmic
        return :red
    elseif schedule == :very_fast
        return :yellow
    end
    :black
end

function make_method_image(problem::SAProblem, result, file::String)
    if problem.dim == 1
        xx = range(problem.a, problem.b, length = 800)
        yy = problem.f.(xx)
        c = schedule_color(result.schedule)
        p = plot(
            xx,
            yy,
            color = :black,
            linewidth = 2,
            label = "f(x)",
            xlabel = "x",
            ylabel = "f(x)",
            title = "$(problem.name): $(schedule_label(result.schedule))",
            legend = :topright
        )
        plot!(p, result.x_hist, result.f_hist, color = c, linewidth = 2, label = "trajectory")
        scatter!(p, [result.x_hist[1]], [result.f_hist[1]], color = :green, markerstrokecolor = :green, markersize = 7, label = "start")
        scatter!(p, [result.x_hist[end]], [result.f_hist[end]], color = c, markerstrokecolor = c, markersize = 8, label = "end")
        savefig(p, file)
        return
    end

    c = schedule_color(result.schedule)
    hx = [pt[1] for pt in result.x_hist]
    hy = [pt[2] for pt in result.x_hist]
    gx = range(problem.a, problem.b, length = 220)
    gy = range(problem.a, problem.b, length = 220)
    gz = [problem.f((x1, x2)) for x2 in gy, x1 in gx]
    p = contour(
        gx,
        gy,
        gz,
        levels = 30,
        c = :grays,
        linewidth = 1,
        colorbar = false,
        xlabel = "x1",
        ylabel = "x2",
        title = "$(problem.name): $(schedule_label(result.schedule))",
        legend = :topright,
        aspect_ratio = :equal
    )
    plot!(p, hx, hy, color = c, linewidth = 2, label = "trajectory")
    scatter!(p, [hx[1]], [hy[1]], color = :green, markerstrokecolor = :green, markersize = 7, label = "start")
    scatter!(p, [hx[end]], [hy[end]], color = c, markerstrokecolor = c, markersize = 8, label = "end")
    savefig(p, file)
end

function print_result_block(result)
    name = schedule_label(result.schedule)
    if result.best_x isa Tuple
        if result.hit_iter == -1
            @printf("%s: no convergence in %d iterations; time = %.6f s; best x = (%.4f, %.4f); best f(x) = %.6f\n",
                name, length(result.temp_hist), result.hit_time_s, result.best_x[1], result.best_x[2], result.best_f)
        else
            @printf("%s: convergence time = %.6f s; iteration = %d; best x = (%.4f, %.4f); best f(x) = %.6f\n",
                name, result.hit_time_s, result.hit_iter, result.best_x[1], result.best_x[2], result.best_f)
        end
        return
    end

    if result.hit_iter == -1
        @printf("%s: no convergence in %d iterations; time = %.6f s; best x = %.4f; best f(x) = %.6f\n",
            name, length(result.temp_hist), result.hit_time_s, result.best_x, result.best_f)
    else
        @printf("%s: convergence time = %.6f s; iteration = %d; best x = %.4f; best f(x) = %.6f\n",
            name, result.hit_time_s, result.hit_iter, result.best_x, result.best_f)
    end
end

function run_problem(problem::SAProblem, cfg::SAConfig, out_prefix::String)
    schedules = [:geometric, :logarithmic, :very_fast]
    results = [run_sa(problem, s, cfg) for s in schedules]

    println("Function: $(problem.name)")
    for r in results
        print_result_block(r)
        img_file = "$(out_prefix)_$(schedule_slug(r.schedule)).png"
        make_method_image(problem, r, img_file)
    end
end

cfg = SAConfig(
    30000,
    1e-4,
    300.0,
    0.9995,
    0.0030,
    1,
    67
)

problems = [
    SAProblem(
        "Example polynomial",
        poly4,
        0.0,
        7.0,
        0.01,
        4.0,
        :temperature,
        1
    ),
    SAProblem(
        "Method guide function",
        phi_method,
        -5.0,
        5.0,
        1.0,
        4.0,
        :unit,
        1
    ),
    SAProblem(
        "Rastrigin 2D",
        rastrigin2d,
        -5.12,
        5.12,
        0.01,
        (4.0, 4.0),
        :temperature,
        2
    )
]

out_prefixes = [
    joinpath(@__DIR__, "sa_poly"),
    joinpath(@__DIR__, "sa_method"),
    joinpath(@__DIR__, "sa_rastrigin")
]

for (problem, out_prefix) in zip(problems, out_prefixes)
    run_problem(problem, cfg, out_prefix)
end