using Random

schwefel1d(x) = 418.9828872724338 - x * sin(sqrt(abs(x)))
poly4(x) = 5 - 24x + 17x^2 - (11 / 3) * x^3 + (1 / 4) * x^4
phi_method(x) = (x^2 - 4)^2 + 0.5 * x

struct SAProblem
    name::String
    f::Function
    a::Float64
    b::Float64
    h::Float64
    x0::Float64
    neighborhood::Symbol
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
temp_very_fast(t::Int, cfg::SAConfig) = cfg.T0 * exp(-cfg.beta * t^(1 / cfg.dim))

function temperature(schedule::Symbol, t::Int, cfg::SAConfig)
    schedule == :geometric && return temp_geometric(t, cfg)
    schedule == :logarithmic && return temp_logarithmic(t, cfg)
    schedule == :very_fast && return temp_very_fast(t, cfg)
    error("bad")
end

function grid_optimum(problem::SAProblem)
    xs = make_grid(problem.a, problem.b, problem.h)
    ys = problem.f.(xs)
    k = argmin(ys)
    xs[k], ys[k]
end

function run_sa(problem::SAProblem, schedule::Symbol, cfg::SAConfig)
    rng = MersenneTwister(cfg.seed)
    xs = make_grid(problem.a, problem.b, problem.h)
    n = length(xs)
    i = nearest_index(problem.x0, problem.a, problem.h, n)
    x = xs[i]
    y = problem.f(x)

    _, f_opt = grid_optimum(problem)
    best_f = y

    for t in 1:cfg.max_iters
        T = max(temperature(schedule, t, cfg), 1e-12)
        step = if problem.neighborhood == :unit
            rand(rng, Bool) ? 1 : -1
        else
            max_jump = max(1, Int(round(0.2 * (n - 1))))
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
        end

        if abs(best_f - f_opt) <= cfg.eps
            return true, t
        end
    end
    return false, cfg.max_iters
end

problems = [
    SAProblem("poly", poly4, 0.0, 7.0, 0.01, 4.0, :temperature),
    SAProblem("method", phi_method, -5.0, 5.0, 1.0, 4.0, :unit),
    SAProblem("schwefel", schwefel1d, -500.0, 500.0, 1.0, 4.0, :temperature)
]

schedules = [:geometric, :logarithmic, :very_fast]
seeds = [11, 29, 67, 101, 777]

candidates = NamedTuple[]
for it in (2000, 4000, 8000, 12000)
    for T0 in (20.0, 40.0, 80.0, 120.0, 200.0)
        for alpha in (0.90, 0.95, 0.97, 0.985, 0.992, 0.996, 0.998)
            for beta in (0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03)
                conv = 0
                iter_sum = 0
                for seed in seeds
                    cfg = SAConfig(it, 1e-4, T0, alpha, beta, 1, seed)
                    for p in problems
                        for s in schedules
                            ok, ti = run_sa(p, s, cfg)
                            conv += ok ? 1 : 0
                            iter_sum += ti
                        end
                    end
                end
                push!(candidates, (conv=conv, iter_sum=iter_sum, it=it, T0=T0, alpha=alpha, beta=beta))
            end
        end
    end
end

sort!(candidates, by = x -> (-x.conv, x.iter_sum))
for c in candidates[1:15]
    println(c)
end
