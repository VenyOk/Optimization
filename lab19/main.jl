using Random
using Printf
using Plots

function schwefel(x)
    d = length(x)
    418.9829 * d - sum(xi * sin(sqrt(abs(xi))) for xi in x)
end

function rosenbrock(x)
    sum(100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:length(x)-1)
end

function rastrigin(x)
    d = length(x)
    10 * d + sum(xi^2 - 10 * cos(2 * pi * xi) for xi in x)
end

function bits_for_bounds(bounds::Vector{Tuple{Float64, Float64}}, h::Float64)
    [max(1, ceil(Int, log2((b - a) / h + 1))) for (a, b) in bounds]
end

function decode_coord(ch::BitVector, a::Float64, b::Float64, h::Float64)
    v = 0
    for bit in ch
        v = (v << 1) | Int(bit)
    end
    m = (1 << length(ch)) - 1
    x = a + (b - a) * v / m
    a + round((x - a) / h) * h
end

function to_point(ch::BitVector, bounds::Vector{Tuple{Float64, Float64}}, h::Float64, bits::Vector{Int})
    point = Vector{Float64}(undef, length(bounds))
    left = 1
    for i in eachindex(bounds)
        right = left + bits[i] - 1
        a, b = bounds[i]
        point[i] = decode_coord(ch[left:right], a, b, h)
        left = right + 1
    end
    point
end

function pop_points(pop::Vector{BitVector}, bounds::Vector{Tuple{Float64, Float64}}, h::Float64, bits::Vector{Int})
    points = Matrix{Float64}(undef, length(pop), length(bounds))
    for i in eachindex(pop)
        points[i, :] = to_point(pop[i], bounds, h, bits)
    end
    points
end

function eval_population(pop::Vector{BitVector}, f, bounds::Vector{Tuple{Float64, Float64}}, h::Float64, bits::Vector{Int})
    points = pop_points(pop, bounds, h, bits)
    values = [f(points[i, :]) for i in 1:size(points, 1)]
    points, values
end

function pick(y::Vector{Float64}, rng::AbstractRNG)
    fit = maximum(y) .- y .+ 1e-12
    r = rand(rng) * sum(fit)
    s = 0.0
    for i in eachindex(fit)
        s += fit[i]
        if s >= r
            return i
        end
    end
    length(fit)
end

function cross(a::BitVector, b::BitVector, pc::Float64, rng::AbstractRNG)
    c1 = copy(a)
    c2 = copy(b)
    if rand(rng) < pc
        p = rand(rng, 1:(length(a) - 1))
        c1[p+1:end] = b[p+1:end]
        c2[p+1:end] = a[p+1:end]
    end
    c1, c2
end

function mut!(ch::BitVector, pm::Float64, rng::AbstractRNG)
    changed = false
    for i in eachindex(ch)
        if rand(rng) < pm
            ch[i] = !ch[i]
            changed = true
        end
    end
    changed
end

function points_from_list(items::Vector{Vector{Float64}}, d::Int)
    points = Matrix{Float64}(undef, length(items), d)
    for i in eachindex(items)
        points[i, :] = items[i]
    end
    points
end

function ga2d(; f, bounds, h = 0.01, n = 100, iters = 200, pc = 0.8, pm = 0.01, seed = 42)
    rng = MersenneTwister(seed)
    bits = bits_for_bounds(bounds, h)
    d = length(bounds)
    pop = [bitrand(rng, sum(bits)) for _ in 1:n]
    hist = Matrix{Float64}[]
    mut_hist = Matrix{Float64}[]

    points, values = eval_population(pop, f, bounds, h, bits)
    push!(hist, copy(points))
    push!(mut_hist, zeros(0, d))
    k = argmin(values)
    best_point = vec(copy(points[k, :]))
    best_value = values[k]

    for _ in 1:iters
        points, values = eval_population(pop, f, bounds, h, bits)
        k = argmin(values)
        if values[k] < best_value
            best_value = values[k]
            best_point = vec(copy(points[k, :]))
        end

        kids = BitVector[]
        mut_points = Vector{Vector{Float64}}()
        while length(kids) < n
            i = pick(values, rng)
            j = pick(values, rng)
            c1, c2 = cross(pop[i], pop[j], pc, rng)
            m1 = mut!(c1, pm, rng)
            m2 = mut!(c2, pm, rng)
            push!(kids, c1)
            if m1
                push!(mut_points, to_point(c1, bounds, h, bits))
            end
            if length(kids) < n
                push!(kids, c2)
                if m2
                    push!(mut_points, to_point(c2, bounds, h, bits))
                end
            end
        end

        all = vcat(pop, kids)
        _, all_values = eval_population(all, f, bounds, h, bits)
        ord = sortperm(all_values)
        pop = [copy(all[ord[i]]) for i in 1:n]
        push!(hist, pop_points(pop, bounds, h, bits))
        if isempty(mut_points)
            push!(mut_hist, zeros(0, d))
        else
            push!(mut_hist, points_from_list(mut_points, d))
        end
    end

    points, values = eval_population(pop, f, bounds, h, bits)
    k = argmin(values)
    if values[k] < best_value
        best_value = values[k]
        best_point = vec(copy(points[k, :]))
    end

    conv = -1
    for i in eachindex(hist)
        if all(all(abs.(hist[i][j, :] .- best_point) .< 1e-9) for j in 1:size(hist[i], 1))
            conv = i - 1
            break
        end
    end

    best_point, best_value, conv, hist, mut_hist
end

function make_gif(name, f, bounds, hist, mut_hist)
    xs = range(bounds[1][1], bounds[1][2], length = 100)
    ys = range(bounds[2][1], bounds[2][2], length = 100)
    z = [f([x, y]) for y in ys, x in xs]
    anim = @animate for i in eachindex(hist)
        start_pts = i == 1 ? hist[1] : hist[i - 1]
        final_pts = hist[i]
        muts = mut_hist[i]
        p = contour(
            xs,
            ys,
            z,
            levels = 25,
            xlabel = "x1",
            ylabel = "x2",
            title = "$(name), iter $(i - 1)",
            legend = :topright,
            color = :gray
        )
        scatter!(p, start_pts[:, 1], start_pts[:, 2], color = :blue, markerstrokecolor = :blue, markersize = 4, label = "Initial")
        if size(muts, 1) > 0
            scatter!(p, muts[:, 1], muts[:, 2], color = :red, markerstrokecolor = :red, markersize = 3, label = "Mutated")
        end
        scatter!(p, final_pts[:, 1], final_pts[:, 2], color = :black, markerstrokecolor = :black, markersize = 4, label = "Final")
    end
    gif(anim, joinpath(@__DIR__, lowercase(name) * ".gif"), fps = 4)
end

function run_test(name, f, bounds, optimum; h, n, iters, pc, pm, seed)
    best_point, best_value, conv, hist, mut_hist = ga2d(f = f, bounds = bounds, h = h, n = n, iters = iters, pc = pc, pm = pm, seed = seed)
    make_gif(name, f, bounds, hist, mut_hist)
    err = sqrt(sum((best_point .- optimum) .^ 2))
    @printf("%-10s x = (%.6f, %.6f), f(x) = %.6f, err = %.6f, iter = %d\n", name, best_point[1], best_point[2], best_value, err, conv)
end

tests = [
    ("Schwefel", schwefel, [(-500.0, 500.0), (-500.0, 500.0)], [420.9687, 420.9687], 0.1, 160, 250, 0.8, 0.02, 67),
    ("Rosenbrock", rosenbrock, [(-2.0, 2.0), (-1.0, 3.0)], [1.0, 1.0], 0.001, 120, 180, 0.8, 0.02, 67),
    ("Rastrigin", rastrigin, [(-5.12, 5.12), (-5.12, 5.12)], [0.0, 0.0], 0.01, 160, 220, 0.8, 0.02, 67)
]

for (name, f, bounds, optimum, h, n, iters, pc, pm, seed) in tests
    run_test(name, f, bounds, optimum; h = h, n = n, iters = iters, pc = pc, pm = pm, seed = seed)
end
