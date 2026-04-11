using Random
using Printf
const _plots_available = try
    @eval using Plots
    true
catch
    false
end

function _require_plots()
    if !_plots_available
        error("Для построения графиков нужен пакет `Plots.jl`. Установите, например: julia -e 'using Pkg; Pkg.add(\"Plots\")'")
    end
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

function euclidean_distance_matrix(points::Vector{Tuple{Float64, Float64}})
    n = length(points)
    dist = fill(Inf, n, n)
    for i in 1:n
        xi, yi = points[i]
        for j in i + 1:n
            xj, yj = points[j]
            dij = hypot(xi - xj, yi - yj)
            dist[i, j] = dij
            dist[j, i] = dij
        end
    end
    dist
end

function tsplib_euc2d_distance_matrix(points::Vector{Tuple{Float64, Float64}})
    n = length(points)
    dist = fill(Inf, n, n)
    for i in 1:n
        xi, yi = points[i]
        for j in i + 1:n
            xj, yj = points[j]
            dij = floor(hypot(xi - xj, yi - yj) + 0.5)
            dist[i, j] = dij
            dist[j, i] = dij
        end
    end
    dist
end

function route_length(route::Vector{Int}, dist::AbstractMatrix{<:Real})
    total = 0.0
    n = length(route)
    for i in 1:n - 1
        total += dist[route[i], route[i + 1]]
    end
    total + dist[route[end], route[1]]
end

function estimate_tau0(dist::Matrix{Float64})
    n = size(dist, 1)
    s = 0.0
    c = 0
    for i in 1:n
        for j in i + 1:n
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
    vcat(route[idx:end], route[1:idx - 1])
end

function canonical_cycle(route::Vector{Int}, start_city::Int = 1)
    forward = rotate_to_start(route, start_city)
    backward = rotate_to_start(reverse(route), start_city)
    length(forward) <= 1 && return forward
    backward[2] < forward[2] ? backward : forward
end

function two_opt!(route::Vector{Int}, dist::Matrix{Float64}; max_swaps::Int = 50)
    n = length(route)
    swaps = 0
    improved = true
    while improved && swaps < max_swaps
        improved = false
        for i in 2:n-1
            for k in i+1:n
                a = route[i - 1]
                b = route[i]
                c = route[k]
                d = k == n ? route[1] : route[k + 1]
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
        current = route[pos - 1]
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
    for i in 1:length(route) - 1
        a = route[i]
        b = route[i + 1]
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

    initial_mean_length = sum(initial_lengths) / length(initial_lengths)
    best_route = copy(initial_best_route)
    best_length = initial_best_length
    best_history = Float64[best_length]
    mean_history = Float64[initial_mean_length]
    pheromone_history = Matrix{Float64}[copy(pheromone)]
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

        push!(best_history, best_length)
        push!(mean_history, sum(lengths) / length(lengths))
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
        push!(pheromone_history, copy(pheromone))
    end
    (
        route = canonical_cycle(best_route, 1),
        length = best_length,
        initial_route = canonical_cycle(initial_best_route, 1),
        initial_length = initial_best_length,
        best_history = best_history,
        mean_history = mean_history,
        pheromone_history = pheromone_history,
    )
end

function exact_tsp_bruteforce(dist::Matrix{Float64})
    n = size(dist, 1)
    visited = falses(n)
    visited[1] = true
    current_route = Vector{Int}(undef, n)
    current_route[1] = 1
    best_route = Ref(collect(1:n))
    best_length = Ref(Inf)
    function dfs(depth::Int, current::Int, total::Float64)
        if total >= best_length[]
            return
        end
        if depth > n
            candidate = total + dist[current, 1]
            if candidate < best_length[]
                best_length[] = candidate
                best_route[] = copy(current_route)
            end
            return
        end
        for city in 2:n
            if !visited[city]
                visited[city] = true
                current_route[depth] = city
                dfs(depth + 1, city, total + dist[current, city])
                visited[city] = false
            end
        end
    end
    dfs(2, 1, 0.0)
    (route = best_route[], length = best_length[])
end

function write_points_table(path::AbstractString, points::Vector{Tuple{Float64, Float64}})
    open(path, "w") do io
        for (idx, point) in enumerate(points)
            println(io, idx, '\t', point[1], '\t', point[2])
        end
    end
end

function write_route_table(path::AbstractString, route::Vector{Int})
    open(path, "w") do io
        for city in route
            println(io, city)
        end
    end
end

function write_history_table(path::AbstractString, best_history::Vector{Float64}, mean_history::Vector{Float64})
    open(path, "w") do io
        for generation in eachindex(best_history)
            println(io, generation - 1, '\t', mean_history[generation], '\t', best_history[generation])
        end
    end
end

function render_route_png(output_path::AbstractString, points::Vector{Tuple{Float64, Float64}}, route::Vector{Int}; title::AbstractString, xlabel::AbstractString, ylabel::AbstractString, show_labels::Bool, show_all_edges::Bool)
    _require_plots()
    Plots.gr()
    try
        Plots.default(fontfamily = "DejaVu Sans")
    catch
        nothing
    end

    mkpath(dirname(output_path))

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    route_closed = vcat(route, route[1])
    rx = [points[i][1] for i in route_closed]
    ry = [points[i][2] for i in route_closed]

    plt = Plots.plot(
        rx,
        ry;
        seriestype = :path,
        linewidth = 2,
        label = "Маршрут",
        xlabel = xlabel,
        ylabel = ylabel,
        title = title,
    )

    if show_all_edges
        n = length(points)
        for i in 1:n-1
            xi, yi = points[i]
            for j in i+1:n
                xj, yj = points[j]
                Plots.plot!(plt, [xi, xj], [yi, yj]; color = :gray, alpha = 0.25, linewidth = 0.6, label = false)
            end
        end
        Plots.plot!(plt, rx, ry; seriestype = :path, linewidth = 2.4, color = :red, label = "Маршрут")
    end

    Plots.plot!(
        plt,
        xs,
        ys;
        seriestype = :scatter,
        markersize = 6,
        color = :black,
        label = "Вершины",
        markerstrokewidth = 0.5,
    )

    if show_labels
        for i in eachindex(points)
            Plots.annotate!(plt, (xs[i], ys[i]), Plots.text(string(i), 8, :black))
        end
    end

    redirect_stderr(devnull) do
        Plots.savefig(plt, output_path)
    end
    return output_path
end

function render_convergence_png(output_path::AbstractString, best_history::Vector{Float64}, mean_history::Vector{Float64}; title::AbstractString)
    _require_plots()
    Plots.gr()

    try
        Plots.default(fontfamily = "DejaVu Sans")
    catch
        nothing
    end

    mkpath(dirname(output_path))

    n = min(length(best_history), length(mean_history))
    x = collect(0:n - 1)

    plt = Plots.plot(
        x,
        mean_history[1:n];
        linewidth = 2,
        label = "Среднее расстояние",
        xlabel = "Поколение",
        ylabel = "Расстояние",
        title = title,
    )

    Plots.plot!(
        plt,
        x,
        best_history[1:n];
        linewidth = 2,
        label = "Кратчайшее расстояние",
        linestyle = :dash,
    )

    redirect_stderr(devnull) do
        Plots.savefig(plt, output_path)
    end
    return output_path
end

function positive_offdiag_values(matrix::Matrix{Float64})
    values = Float64[]
    n = size(matrix, 1)
    for i in 1:n-1
        for j in i+1:n
            value = matrix[i, j]
            if isfinite(value) && value > 0.0
                push!(values, value)
            end
        end
    end
    values
end

function render_pheromone_animation(output_path::AbstractString, points::Vector{Tuple{Float64, Float64}}, pheromone_history::Vector{Matrix{Float64}}; title::AbstractString, xlabel::AbstractString, ylabel::AbstractString)
    _require_plots()
    Plots.gr()

    try
        Plots.default(fontfamily = "DejaVu Sans")
    catch
        nothing
    end

    mkpath(dirname(output_path))

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    n = length(points)
    all_values = Float64[]
    for matrix in pheromone_history
        append!(all_values, positive_offdiag_values(matrix))
    end
    min_tau = isempty(all_values) ? 0.0 : minimum(all_values)
    max_tau = isempty(all_values) ? 1.0 : maximum(all_values)
    span_tau = max(max_tau - min_tau, eps(Float64))

    anim = Plots.Animation()
    for (frame_index, pheromone) in enumerate(pheromone_history)
        plt = Plots.plot(
            xs,
            ys;
            seriestype = :scatter,
            markersize = 4,
            color = :black,
            label = false,
            xlabel = xlabel,
            ylabel = ylabel,
            title = string(title, " | поколение ", frame_index - 1),
            legend = false,
        )
        for i in 1:n-1
            xi, yi = points[i]
            for j in i+1:n
                tau = pheromone[i, j]
                if !(isfinite(tau) && tau > 0.0)
                    continue
                end
                strength = (tau - min_tau) / span_tau
                alpha = 0.08 + 0.92 * strength
                width = 0.3 + 5.0 * strength
                color = RGBA(0.75 - 0.55 * strength, 0.75 - 0.55 * strength, 0.8, alpha)
                xj, yj = points[j]
                Plots.plot!(plt, [xi, xj], [yi, yj]; color = color, linewidth = width, label = false)
            end
        end
        Plots.scatter!(plt, xs, ys; markersize = 4.5, color = :black, markerstrokewidth = 0.3, label = false)
        Plots.frame(anim, plt)
    end

    redirect_stderr(devnull) do
        Plots.gif(anim, output_path; fps = 2)
    end
    return output_path
end

function join_route(route::Vector{Int})
    join(vcat(route, route[1]), " -> ")
end

function berlin52_points()
    [
        (565.0, 575.0),
        (25.0, 185.0),
        (345.0, 750.0),
        (945.0, 685.0),
        (845.0, 655.0),
        (880.0, 660.0),
        (25.0, 230.0),
        (525.0, 1000.0),
        (580.0, 1175.0),
        (650.0, 1130.0),
        (1605.0, 620.0),
        (1220.0, 580.0),
        (1465.0, 200.0),
        (1530.0, 5.0),
        (845.0, 680.0),
        (725.0, 370.0),
        (145.0, 665.0),
        (415.0, 635.0),
        (510.0, 875.0),
        (560.0, 365.0),
        (300.0, 465.0),
        (520.0, 585.0),
        (480.0, 415.0),
        (835.0, 625.0),
        (975.0, 580.0),
        (1215.0, 245.0),
        (1320.0, 315.0),
        (1250.0, 400.0),
        (660.0, 180.0),
        (410.0, 250.0),
        (420.0, 555.0),
        (575.0, 665.0),
        (1150.0, 1160.0),
        (700.0, 580.0),
        (685.0, 595.0),
        (685.0, 610.0),
        (770.0, 610.0),
        (795.0, 645.0),
        (720.0, 635.0),
        (760.0, 650.0),
        (475.0, 960.0),
        (95.0, 260.0),
        (875.0, 920.0),
        (700.0, 500.0),
        (555.0, 815.0),
        (830.0, 485.0),
        (1170.0, 65.0),
        (830.0, 610.0),
        (605.0, 625.0),
        (595.0, 360.0),
        (1340.0, 725.0),
        (1740.0, 245.0),
    ]
end

function berlin52_optimal_route()
    [
        1, 49, 32, 45, 19, 41, 8, 9, 10, 43, 33, 51, 11, 52, 14, 13, 47, 26,
        27, 28, 12, 25, 4, 6, 15, 5, 24, 48, 38, 37, 40, 39, 36, 35, 34, 44,
        46, 16, 29, 50, 20, 23, 30, 2, 7, 42, 21, 17, 3, 18, 31, 22,
    ]
end

function demo_complete_graph_points()
    [
        (0.0, 0.0),
        (20.0, 5.0),
        (35.0, 0.0),
        (42.0, 28.0),
        (28.0, 45.0),
        (6.0, 32.0),
        (12.0, 18.0),
        (30.0, 17.0),
        (18.0, 27.0),
    ]
end

function run_complete_graph_case(output_dir::AbstractString)
    points = demo_complete_graph_points()
    dist = euclidean_distance_matrix(points)
    exact = exact_tsp_bruteforce(dist)
    tau0 = estimate_tau0(dist)
    params = ACOParams(length(points), 180, 1.0, 5.0, 0.5, 120.0, tau0, 0, true, true, 25)
    result = ant_system_tsp(dist; params = params)
    route_path = joinpath(output_dir, "полный_граф_маршрут.png")
    convergence_path = joinpath(output_dir, "полный_граф_сходимость.png")
    render_route_png(
        route_path,
        points,
        result.route;
        title = "Полносвязный граф: лучший найденный маршрут",
        xlabel = "Координата X",
        ylabel = "Координата Y",
        show_labels = true,
        show_all_edges = true,
    )
    render_convergence_png(
        convergence_path,
        result.best_history,
        result.mean_history;
        title = "Полносвязный граф: сходимость муравьиной системы",
    )
    gap = 100 * (result.length - exact.length) / exact.length
    println("Полносвязный граф")
    @printf("  Число вершин: %d\n", length(points))
    @printf("  Длина маршрута муравьиной системы: %.3f\n", result.length)
    @printf("  Точная длина: %.3f\n", exact.length)
    @printf("  Относительный зазор: %.3f%%\n", gap)
    println()
end

function weighted_distance_matrix(points::Vector{Tuple{Float64, Float64}}, weights::Vector{Float64})
    n = length(points)
    dist = fill(Inf, n, n)
    for i in 1:n-1
        xi, yi = points[i]
        for j in i+1:n
            xj, yj = points[j]
            base = hypot(xi - xj, yi - yj)
            w = (weights[i] + weights[j]) / 2
            cost = base * w
            dist[i, j] = cost
            dist[j, i] = cost
        end
    end
    dist
end

function run_random10_weighted_case(output_dir::AbstractString)
    n = 10
    rng = MersenneTwister()
    points = [(rand(rng) * 1000, rand(rng) * 1000) for _ in 1:n]
    weights = [0.5 + 1.5 * rand(rng) for _ in 1:n]
    dist = weighted_distance_matrix(points, weights)
    exact = exact_tsp_bruteforce(dist)
    tau0 = estimate_tau0(dist)
    params = ACOParams(50, 400, 1.0, 5.0, 0.5, 150.0, tau0, 6, true, false, 0)
    result = ant_system_tsp(dist; params = params)
    route_path = joinpath(output_dir, "random10_маршрут.png")
    convergence_path = joinpath(output_dir, "random10_сходимость.png")
    render_route_png(
        route_path,
        points,
        result.route;
        title = "Случайные 10 точек (веса): лучший маршрут",
        xlabel = "X",
        ylabel = "Y",
        show_labels = false,
        show_all_edges = true,
    )
    render_convergence_png(
        convergence_path,
        result.best_history,
        result.mean_history;
        title = "Случайные 10 точек (веса): сходимость",
    )
    gap = 100 * (result.length - exact.length) / exact.length
    println("Случайные 10 точек с весами")
    @printf("  Лучший найденный маршрут: %.3f\n", result.length)
    @printf("  Эталонная длина: %.3f\n", exact.length)
    @printf("  Отклонение: %.3f%%\n", gap)
    println()
end

function run_berlin52_case(output_dir::AbstractString)
    points = berlin52_points()
    dist = tsplib_euc2d_distance_matrix(points)
    params = ACOParams(53, 10, 1.0, 5.0, 0.9, 20.0, 1e-6, 2, true, true, 7)
    result = ant_system_tsp(dist; params = params)
    known_best_route = berlin52_optimal_route()
    known_best_length = route_length(known_best_route, dist)
    initial_path = joinpath(output_dir, "berlin52_начальный_маршрут.png")
    convergence_path = joinpath(output_dir, "berlin52_сходимость.png")
    best_path = joinpath(output_dir, "berlin52_лучший_маршрут.png")
    optimal_path = joinpath(output_dir, "berlin52_эталонный_маршрут.png")
    pheromone_gif_path = joinpath(output_dir, "berlin52_феромоны.gif")
    render_route_png(
        initial_path,
        points,
        result.initial_route;
        title = "Berlin52: лучший маршрут начальной случайной популяции",
        xlabel = "Долгота",
        ylabel = "Широта",
        show_labels = false,
        show_all_edges = false,
    )
    render_convergence_png(
        convergence_path,
        result.best_history,
        result.mean_history;
        title = "Berlin52: сходимость муравьиной системы",
    )
    render_route_png(
        best_path,
        points,
        result.route;
        title = "Berlin52: лучший маршрут после 10 поколений",
        xlabel = "Долгота",
        ylabel = "Широта",
        show_labels = false,
        show_all_edges = false,
    )
    render_route_png(
        optimal_path,
        points,
        known_best_route;
        title = "Berlin52: эталонный маршрут",
        xlabel = "Долгота",
        ylabel = "Широта",
        show_labels = false,
        show_all_edges = false,
    )
    render_pheromone_animation(
        pheromone_gif_path,
        points,
        result.pheromone_history;
        title = "Berlin52: эволюция феромонов",
        xlabel = "Долгота",
        ylabel = "Широта",
    )
    gap = 100 * (result.length - known_best_length) / known_best_length
    println("Berlin52")
    @printf("  Число вершин: %d\n", length(points))
    @printf("  Число муравьев: %d\n", params.ants)
    @printf("  Число поколений: %d\n", params.iterations)
    @printf("  Лучший найденный маршрут: %.0f\n", result.length)
    @printf("  Эталонная длина: %.0f\n", known_best_length)
    @printf("  Отклонение от эталона: %.3f%%\n", gap)
    println()
end

function main()
    output_dir = joinpath(@__DIR__, "output")
    mkpath(output_dir)
    run_complete_graph_case(output_dir)
    run_random10_weighted_case(output_dir)
    run_berlin52_case(output_dir)
end

main()
