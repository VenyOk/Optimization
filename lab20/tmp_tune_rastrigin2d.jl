using Random

rastrigin2d(v::NTuple{2, Float64}) = 20 + v[1]^2 + v[2]^2 - 10 * (cos(2 * pi * v[1]) + cos(2 * pi * v[2]))

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

function temp(schedule::Symbol, t::Int, T0::Float64, alpha::Float64, beta::Float64)
    schedule == :geometric && return T0 * alpha^(t - 1)
    schedule == :logarithmic && return T0 / log(t + 1)
    schedule == :very_fast && return T0 * exp(-beta * t^(1 / 2))
    error("bad schedule")
end

function run_one(schedule::Symbol, seed::Int, T0::Float64, alpha::Float64, beta::Float64; iters=8000, eps=1e-4)
    rng = MersenneTwister(seed)
    a = -5.12
    b = 5.12
    h = 0.01
    xs = make_grid(a, b, h)
    n = length(xs)

    i1 = nearest_index(4.0, a, h, n)
    i2 = nearest_index(4.0, a, h, n)
    x = (xs[i1], xs[i2])
    y = rastrigin2d(x)
    best_f = y

    max_jump = max(1, Int(round(0.5 * (n - 1))))

    for t in 1:iters
        T = max(temp(schedule, t, T0, alpha, beta), 1e-12)
        jump = Int(round(max_jump * T / T0))
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
        y_new = rastrigin2d(x_new)
        d = y_new - y

        if d <= 0 || rand(rng) < exp(-d / T)
            i1 = j1
            i2 = j2
            x = x_new
            y = y_new
        end

        if y < best_f
            best_f = y
            if best_f <= eps
                return true, t, best_f
            end
        end
    end

    return false, iters, best_f
end

function evaluate_schedule(schedule::Symbol, T0_vals, alpha_vals, beta_vals; seeds=1:40)
    best = (succ=-1, mean_iter=Inf, mean_best=Inf, T0=0.0, alpha=0.0, beta=0.0)
    for T0 in T0_vals
        for alpha in alpha_vals
            for beta in beta_vals
                succ = 0
                iter_sum = 0
                best_sum = 0.0
                for s in seeds
                    ok, it, bf = run_one(schedule, s, T0, alpha, beta)
                    succ += ok ? 1 : 0
                    iter_sum += it
                    best_sum += bf
                end
                mean_iter = iter_sum / length(seeds)
                mean_best = best_sum / length(seeds)
                cand = (succ=succ, mean_iter=mean_iter, mean_best=mean_best, T0=T0, alpha=alpha, beta=beta)
                if cand.succ > best.succ || (cand.succ == best.succ && cand.mean_iter < best.mean_iter)
                    best = cand
                end
            end
        end
    end
    best
end

println("Tuning geometric...")
best_geo = evaluate_schedule(:geometric, [20.0, 50.0, 100.0, 200.0, 400.0], [0.99, 0.995, 0.997, 0.998, 0.999], [0.0008])
println(best_geo)

println("Tuning logarithmic...")
best_log = evaluate_schedule(:logarithmic, [10.0, 20.0, 40.0, 80.0, 120.0, 200.0], [0.995], [0.0008])
println(best_log)

println("Tuning very_fast...")
best_vf = evaluate_schedule(:very_fast, [20.0, 50.0, 100.0, 200.0, 400.0], [0.995], [0.0001, 0.0002, 0.0004, 0.0008, 0.0015, 0.003, 0.005])
println(best_vf)
