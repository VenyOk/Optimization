using Random
using Printf
using Plots
using Statistics

const _plots_available = true

struct SAParams
    T0::Float64
    Tmin::Float64
    tmax::Int
    cooling::Function
end

function berlin52_coords()
    return [
        (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0), (845.0, 655.0),
        (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0), (580.0, 1175.0), (650.0, 1130.0),
        (1605.0, 620.0), (1220.0, 580.0), (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0),
        (725.0, 370.0), (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
        (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0), (975.0, 580.0),
        (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0), (660.0, 180.0), (410.0, 250.0),
        (420.0, 555.0), (575.0, 665.0), (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0),
        (685.0, 610.0), (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
        (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0), (555.0, 815.0),
        (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0), (605.0, 625.0), (595.0, 360.0),
        (1340.0, 725.0), (1740.0, 245.0),
    ]
end

function dist_matrix(coords)
    n = length(coords)
    D = Matrix{Float64}(undef, n, n)
    for i in 1:n
        xi, yi = coords[i]
        for j in 1:n
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            D[i, j] = sqrt(dx * dx + dy * dy)
        end
    end
    return D
end

function tour_length(tour::Vector{Int}, D::Matrix{Float64})
    n = length(tour)
    s = 0.0
    for k in 1:(n - 1)
        s += D[tour[k], tour[k + 1]]
    end
    s += D[tour[end], tour[1]]
    return s
end

function neighbor_2opt(tour::Vector{Int}, rng)
    n = length(tour)
    i = rand(rng, 1:(n - 1))
    j = rand(rng, (i + 1):n)
    if i == 1 && j == n
        return copy(tour)
    end
    x = copy(tour)
    x[i:j] = reverse(x[i:j])
    return x
end

function sa_minimize(x0, phi, neighbor, params::SAParams; rng=Random.default_rng())
    x = copy(x0)
    fx = phi(x)
    bestx = copy(x)
    bestfx = fx
    T = params.T0
    t = 1
    history_length = Float64[]
    history_temp = Float64[]
    start_time = time()
    while t <= params.tmax && T > params.Tmin
        x′ = neighbor(x, rng)
        fx′ = phi(x′)
        Δ = fx′ - fx
        if Δ <= 0 || rand(rng) < exp(-Δ / T)
            x = x′
            fx = fx′
            if fx < bestfx
                bestx = copy(x)
                bestfx = fx
            end
        end
        push!(history_length, bestfx)
        push!(history_temp, T)
        T = params.cooling(params.T0, T, t)
        t += 1
    end
    elapsed = time() - start_time
    return bestx, bestfx, t-1, elapsed, history_length, history_temp
end

function plot_tour(coords, tour, filename)
    x = [coords[i][1] for i in tour]
    y = [coords[i][2] for i in tour]
    push!(x, x[1])
    push!(y, y[1])
    plt = plot(x, y, marker=:circle, line=:path, markersize=5, linewidth=1.5, color=:blue,
               legend=false, title="Tour length: $(round(tour_length(tour, dist_matrix(coords)), digits=2))",
               xlabel="X", ylabel="Y")
    for (i, (xi, yi)) in enumerate(coords)
        annotate!(xi, yi, text(string(i), :left, 7))
    end
    savefig(plt, filename)
end

function geometric_cooling(T0, T_prev, t)
    alpha = 0.9995
    return alpha * T_prev
end

function logarithmic_cooling(T0, T_prev, t)
    return T0 / log(t + 1)
end

function veryfast_cooling(T0, T_prev, t)
    beta = 2.5
    n = 52
    return T0 * exp(-beta * t^(1/n))
end

function run_sa_with_cooling(name, cooling_func, T0, Tmin, tmax)
    rng = MersenneTwister(42)
    coords = berlin52_coords()
    D = dist_matrix(coords)
    n = length(coords)
    x0 = collect(1:n)
    shuffle!(rng, x0)
    phi = x -> tour_length(x, D)
    neighbor = (x, r) -> neighbor_2opt(x, r)
    params = SAParams(T0, Tmin, tmax, cooling_func)
    bestx, bestfx, iters, elapsed, hist_len, hist_temp = sa_minimize(x0, phi, neighbor, params; rng=rng)
    @printf("%-12s | distance = %8.2f | iterations = %7d | time = %6.2f s\n", name, bestfx, iters, elapsed)
    plot_tour(coords, bestx, "tour_$(name).png")
    p1 = plot(hist_len, xlabel="Iteration", ylabel="Best tour length", title="$(name) convergence", legend=false)
    p2 = plot(hist_temp, xlabel="Iteration", ylabel="Temperature", title="$(name) cooling schedule", legend=false)
    p = plot(p1, p2, layout=(2,1), size=(800,600))
    savefig(p, "convergence_$(name).png")
    return (name=name, distance=bestfx, iterations=iters, time=elapsed)
end

struct ACOParams
    ants::Int
    iterations::Int
    alpha::Float64
    beta::Float64
    rho::Float64
    q::Float64
    tau0::Float64
    elite_ants::Int
    random_initialization::Bool
    use_two_opt::Bool
    two_opt_max_swaps::Int
end

function tsplib_euc2d_distance_matrix(points::Vector{Tuple{Float64, Float64}})
    n = length(points)
    dist = fill(Inf, n, n)
    for i in 1:n
        xi, yi = points[i]
        for j in i+1:n
            xj, yj = points[j]
            dij = floor(hypot(xi - xj, yi - yj) + 0.5)
            dist[i, j] = dij
            dist[j, i] = dij
        end
    end
    dist
end

function estimate_tau0(dist::Matrix{Float64})
    n = size(dist, 1)
    s = 0.0
    c = 0
    for i in 1:n
        for j in i+1:n
            dij = dist[i, j]
            if isfinite(dij) && dij > 0.0
                s += dij
                c += 1
            end
        end
    end
    c == 0 ? 1.0 : (c / s)
end

function rotate_to_start(route::Vector{Int}, start_city::Int)
    idx = findfirst(==(start_city), route)
    idx === nothing && return copy(route)
    vcat(route[idx:end], route[1:idx-1])
end

function canonical_cycle(route::Vector{Int}, start_city::Int=1)
    forward = rotate_to_start(route, start_city)
    backward = rotate_to_start(reverse(route), start_city)
    length(forward) <= 1 && return forward
    backward[2] < forward[2] ? backward : forward
end

function two_opt!(route::Vector{Int}, dist::Matrix{Float64}; max_swaps::Int=50)
    n = length(route)
    swaps = 0
    improved = true
    while improved && swaps < max_swaps
        improved = false
        for i in 2:n-1
            for k in i+1:n
                a = route[i-1]
                b = route[i]
                c = route[k]
                d = k == n ? route[1] : route[k+1]
                delta = dist[a, c] + dist[b, d] - (dist[a, b] + dist[c, d])
                if delta < -1e-12
                    route[i:k] = reverse(route[i:k])
                    swaps += 1
                    improved = true
                    break
                end
            end
            if improved
                break
            end
        end
    end
    route
end

function random_tour(n::Int, rng::AbstractRNG)
    route = collect(1:n)
    shuffle!(rng, route)
    route
end

function roulette_choice(cities::Vector{Int}, weights::Vector{Float64}, rng::AbstractRNG)
    total = sum(weights)
    if !(total > 0.0) || !isfinite(total)
        return cities[rand(rng, eachindex(cities))]
    end
    threshold = rand(rng) * total
    acc = 0.0
    for i in eachindex(cities)
        acc += weights[i]
        if threshold <= acc
            return cities[i]
        end
    end
    cities[end]
end

function construct_tour(start::Int, pheromone::Matrix{Float64}, eta::Matrix{Float64}, params::ACOParams, rng::AbstractRNG)
    n = size(pheromone, 1)
    visited = falses(n)
    visited[start] = true
    route = Vector{Int}(undef, n)
    route[1] = start
    for pos in 2:n
        current = route[pos-1]
        candidates = Int[]
        weights = Float64[]
        for city in 1:n
            if !visited[city]
                push!(candidates, city)
                push!(weights, pheromone[current, city]^params.alpha * eta[current, city]^params.beta)
            end
        end
        next_city = roulette_choice(candidates, weights, rng)
        visited[next_city] = true
        route[pos] = next_city
    end
    route
end

function deposit_pheromone!(pheromone::Matrix{Float64}, route::Vector{Int}, delta::Float64)
    for i in 1:length(route)-1
        a = route[i]
        b = route[i+1]
        pheromone[a, b] += delta
        pheromone[b, a] += delta
    end
    a = route[end]
    b = route[1]
    pheromone[a, b] += delta
    pheromone[b, a] += delta
end

function ant_system_tsp(dist::Matrix{Float64}; params::ACOParams)
    n = size(dist, 1)
    eta = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:n
            if i != j && isfinite(dist[i, j]) && dist[i, j] > 0.0
                eta[i, j] = 1.0 / dist[i, j]
            end
        end
    end
    pheromone = fill(params.tau0, n, n)
    for i in 1:n
        pheromone[i, i] = 0.0
    end
    rng = MersenneTwister()
    initial_tours = Vector{Vector{Int}}(undef, params.ants)
    initial_lengths = Vector{Float64}(undef, params.ants)
    for ant in 1:params.ants
        tour = params.random_initialization ? random_tour(n, rng) : construct_tour(rand(rng, 1:n), pheromone, eta, params, rng)
        if params.use_two_opt
            two_opt!(tour, dist; max_swaps = params.two_opt_max_swaps)
        end
        initial_tours[ant] = tour
        initial_lengths[ant] = route_length(tour, dist)
    end
    initial_best_index = argmin(initial_lengths)
    initial_best_route = copy(initial_tours[initial_best_index])
    initial_best_length = initial_lengths[initial_best_index]

    best_route = copy(initial_best_route)
    best_length = initial_best_length
    tours = Vector{Vector{Int}}(undef, params.ants)
    lengths = Vector{Float64}(undef, params.ants)
    for _ in 1:params.iterations
        for ant in 1:params.ants
            start = rand(rng, 1:n)
            tour = construct_tour(start, pheromone, eta, params, rng)
            if params.use_two_opt
                two_opt!(tour, dist; max_swaps = params.two_opt_max_swaps)
            end
            length_tour = route_length(tour, dist)
            tours[ant] = tour
            lengths[ant] = length_tour
            if length_tour < best_length
                best_length = length_tour
                best_route = copy(tour)
            end
        end
        pheromone .*= 1.0 - params.rho
        pheromone .= max.(pheromone, eps(Float64))
        for ant in 1:params.ants
            deposit_pheromone!(pheromone, tours[ant], params.q / lengths[ant])
        end
        if params.elite_ants > 0
            deposit_pheromone!(pheromone, best_route, params.elite_ants * params.q / best_length)
        end
        for i in 1:n
            pheromone[i, i] = 0.0
        end
    end
    return (route = canonical_cycle(best_route, 1), length = best_length)
end

function route_length(route::Vector{Int}, dist::AbstractMatrix{<:Real})
    total = 0.0
    n = length(route)
    for i in 1:n-1
        total += dist[route[i], route[i+1]]
    end
    total + dist[route[end], route[1]]
end

function berlin52_optimal_route()
    [1, 49, 32, 45, 19, 41, 8, 9, 10, 43, 33, 51, 11, 52, 14, 13, 47, 26,
     27, 28, 12, 25, 4, 6, 15, 5, 24, 48, 38, 37, 40, 39, 36, 35, 34, 44,
     46, 16, 29, 50, 20, 23, 30, 2, 7, 42, 21, 17, 3, 18, 31, 22]
end

function run_aco_berlin52()
    points = berlin52_coords()
    dist = tsplib_euc2d_distance_matrix(points)
    tau0 = estimate_tau0(dist)
    params = ACOParams(53, 10, 1.0, 5.0, 0.9, 20.0, tau0, 2, true, true, 7)
    start_time = time()
    result = ant_system_tsp(dist; params = params)
    elapsed = time() - start_time
    known_best_length = route_length(berlin52_optimal_route(), dist)
    gap = 100 * (result.length - known_best_length) / known_best_length

    plot_tour(points, result.route, "tour_ACO.png")

    return (name="ACO", distance=result.length, iterations=params.iterations, time=elapsed, gap=gap)
end

function print_comparison(sa_results, aco_result)

    println(rpad("Метод", 16) * rpad("Расстояние", 14) * rpad("Итерации", 12) * rpad("Время (с)", 12) * rpad("Отклонение", 14))
    println("-"^70)
    for r in sa_results
        opt_dist = 7542.0
        gap = 100 * (r.distance - opt_dist) / opt_dist
        println(rpad(r.name, 16) * @sprintf("%12.2f", r.distance) * @sprintf("%10d", r.iterations) * @sprintf("%12.2f", r.time) * @sprintf("%10.2f%%", gap))
    end
    aco_gap = aco_result.gap
    println(rpad(aco_result.name, 16) * @sprintf("%12.2f", aco_result.distance) * @sprintf("%10d", aco_result.iterations) * @sprintf("%12.2f", aco_result.time) * @sprintf("%10.2f%%", aco_gap))
    opt_dist = 7542.0
    println("Оптимальное известное расстояние: $opt_dist")
end

function main()
    sa_results = []
    push!(sa_results, run_sa_with_cooling("Геометрический", geometric_cooling, 500.0, 1e-3, 250_000))
    push!(sa_results, run_sa_with_cooling("Логарифмический", logarithmic_cooling, 500.0, 1e-3, 250_000))
    push!(sa_results, run_sa_with_cooling("Сверхбыстрый", veryfast_cooling, 500.0, 1e-3, 250_000))
    
    aco_result = run_aco_berlin52()
    
    print_comparison(sa_results, aco_result)
end

main()