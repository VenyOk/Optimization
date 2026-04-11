using Random
using Printf
using LinearAlgebra

const _plots_available = try
    @eval using Plots
    true
catch
    false
end

function _require_plots()
    if !_plots_available
        error("Для построения графиков нужен пакет `Plots.jl`. Установите его, например: julia -e 'using Pkg; Pkg.add(\"Plots\")'")
    end
end

mutable struct Particle
    position::Vector{Float64}
    velocity::Vector{Float64}
    best_position::Vector{Float64}
    best_value::Float64
end

struct ParticleSwarmParams
    swarm_size::Int
    iterations::Int
    phi_p::Float64
    phi_s::Float64
    velocity_scale::Float64
end

struct FishSchoolParams
    school_size::Int
    iterations::Int
    individual_step_initial::Float64
    individual_step_final::Float64
    volitive_step_initial::Float64
    volitive_step_final::Float64
    weight_min::Float64
    weight_max::Float64
end

struct AlgorithmResult
    best_position::Vector{Float64}
    best_value::Float64
    best_history::Vector{Float64}
    mean_history::Vector{Float64}
    positions_history::Vector{Matrix{Float64}}
    iterations::Int
end

struct FunctionSpec
    name::String
    slug::String
    objective::Function
    lower::Vector{Float64}
    upper::Vector{Float64}
    pso_params::ParticleSwarmParams
    fish_params::FishSchoolParams
end

function schwefel(x::AbstractVector{<:Real})
    n = length(x)
    418.9829 * n - sum(xi * sin(sqrt(abs(xi))) for xi in x)
end

function rosenbrock(x::AbstractVector{<:Real})
    sum(100.0 * (x[i + 1] - x[i]^2)^2 + (1.0 - x[i])^2 for i in 1:length(x) - 1)
end

function rastrigin(x::AbstractVector{<:Real})
    n = length(x)
    10.0 * n + sum(xi^2 - 10.0 * cos(2.0 * pi * xi) for xi in x)
end

function linear_schedule(start_value::Float64, finish_value::Float64, iteration::Int, total_iterations::Int)
    total_iterations <= 1 && return finish_value
    ratio = (iteration - 1) / (total_iterations - 1)
    start_value + ratio * (finish_value - start_value)
end

function random_position(lower::Vector{Float64}, upper::Vector{Float64}, rng::AbstractRNG)
    span = upper .- lower
    [lower[i] + rand(rng) * span[i] for i in eachindex(lower)]
end

function clamp_position(position::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64})
    clamp.(position, lower, upper)
end

function snapshot_positions(positions::Vector{Vector{Float64}})
    reduce(vcat, (reshape(copy(position), 1, :) for position in positions))
end

function initialize_particle(
    objective::Function,
    lower::Vector{Float64},
    upper::Vector{Float64},
    params::ParticleSwarmParams,
    rng::AbstractRNG,
)
    span = upper .- lower
    position = random_position(lower, upper, rng)
    velocity = [(2.0 * rand(rng) - 1.0) * params.velocity_scale * span[i] for i in eachindex(span)]
    value = objective(position)
    Particle(position, velocity, copy(position), value)
end

function particle_swarm_optimization(
    objective::Function,
    lower::Vector{Float64},
    upper::Vector{Float64};
    params::ParticleSwarmParams,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)
    particles = [initialize_particle(objective, lower, upper, params, rng) for _ in 1:params.swarm_size]
    current_values = [objective(particle.position) for particle in particles]

    best_index = argmin(current_values)
    global_best_position = copy(particles[best_index].best_position)
    global_best_value = particles[best_index].best_value
    best_history = Float64[global_best_value]
    mean_history = Float64[sum(current_values) / length(current_values)]
    positions_history = Matrix{Float64}[snapshot_positions([particle.position for particle in particles])]

    for _ in 1:params.iterations
        for particle in particles
            for d in eachindex(particle.position)
                rp = rand(rng)
                rs = rand(rng)
                particle.velocity[d] =
                    particle.velocity[d] +
                    params.phi_p * rp * (particle.best_position[d] - particle.position[d]) +
                    params.phi_s * rs * (global_best_position[d] - particle.position[d])
                particle.position[d] += particle.velocity[d]

                if particle.position[d] < lower[d]
                    particle.position[d] = lower[d]
                    particle.velocity[d] = 0.0
                elseif particle.position[d] > upper[d]
                    particle.position[d] = upper[d]
                    particle.velocity[d] = 0.0
                end
            end

            current_value = objective(particle.position)
            if current_value < particle.best_value
                particle.best_value = current_value
                particle.best_position = copy(particle.position)
                if current_value < global_best_value
                    global_best_value = current_value
                    global_best_position = copy(particle.position)
                end
            end
        end

        for i in eachindex(particles)
            current_values[i] = objective(particles[i].position)
        end
        push!(best_history, global_best_value)
        push!(mean_history, sum(current_values) / length(current_values))
        push!(positions_history, snapshot_positions([particle.position for particle in particles]))
    end

    AlgorithmResult(copy(global_best_position), global_best_value, best_history, mean_history, positions_history, params.iterations)
end

function weighted_barycenter(positions::Vector{Vector{Float64}}, weights::Vector{Float64})
    total_weight = sum(weights)
    center = zeros(Float64, length(positions[1]))
    if total_weight <= 0.0
        for position in positions
            center .+= position
        end
        return center ./ length(positions)
    end
    for i in eachindex(positions)
        center .+= positions[i] .* weights[i]
    end
    center ./ total_weight
end

function fish_school_search(
    objective::Function,
    lower::Vector{Float64},
    upper::Vector{Float64};
    params::FishSchoolParams,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)
    span = upper .- lower
    dimension = length(lower)

    positions = [random_position(lower, upper, rng) for _ in 1:params.school_size]
    values = [objective(position) for position in positions]
    weights = fill((params.weight_min + params.weight_max) / 2, params.school_size)
    displacements = [zeros(Float64, dimension) for _ in 1:params.school_size]
    improvements = zeros(Float64, params.school_size)

    best_index = argmin(values)
    best_position = copy(positions[best_index])
    best_value = values[best_index]
    best_history = Float64[best_value]
    mean_history = Float64[sum(values) / length(values)]
    positions_history = Matrix{Float64}[snapshot_positions(positions)]

    for iteration in 1:params.iterations
        current_individual_step = linear_schedule(
            params.individual_step_initial,
            params.individual_step_final,
            iteration,
            params.iterations,
        )
        current_volitive_step = linear_schedule(
            params.volitive_step_initial,
            params.volitive_step_final,
            iteration,
            params.iterations,
        )

        total_weight_before = sum(weights)
        fill!(improvements, 0.0)

        for fish in 1:params.school_size
            step = (rand(rng, dimension) .* 2 .- 1) .* current_individual_step .* span
            candidate = clamp_position(positions[fish] .+ step, lower, upper)
            candidate_value = objective(candidate)
            if candidate_value < values[fish]
                displacements[fish] = candidate .- positions[fish]
                improvements[fish] = values[fish] - candidate_value
                positions[fish] = candidate
                values[fish] = candidate_value
                if candidate_value < best_value
                    best_value = candidate_value
                    best_position = copy(candidate)
                end
            else
                fill!(displacements[fish], 0.0)
            end
        end

        max_improvement = maximum(improvements)
        if max_improvement > 0.0
            for fish in 1:params.school_size
                weights[fish] = clamp(
                    weights[fish] + improvements[fish] / max_improvement,
                    params.weight_min,
                    params.weight_max,
                )
            end
        end

        total_improvement = sum(improvements)
        if total_improvement > 0.0
            instinctive_direction = zeros(Float64, dimension)
            for fish in 1:params.school_size
                if improvements[fish] > 0.0
                    instinctive_direction .+= displacements[fish] .* improvements[fish]
                end
            end
            instinctive_direction ./= total_improvement
            for fish in 1:params.school_size
                positions[fish] = clamp_position(positions[fish] .+ instinctive_direction, lower, upper)
                values[fish] = objective(positions[fish])
                if values[fish] < best_value
                    best_value = values[fish]
                    best_position = copy(positions[fish])
                end
            end
        end

        barycenter = weighted_barycenter(positions, weights)
        collective_target = 0.75 .* barycenter .+ 0.25 .* best_position
        for fish in 1:params.school_size
            direction = collective_target .- positions[fish]
            direction_norm = norm(direction)
            if direction_norm > 0.0
                step = rand(rng) * current_volitive_step .* span .* (direction ./ direction_norm)
                candidate = clamp_position(positions[fish] .+ step, lower, upper)
                candidate_value = objective(candidate)
                if candidate_value <= values[fish]
                    positions[fish] = candidate
                    values[fish] = candidate_value
                    if candidate_value < best_value
                        best_value = candidate_value
                        best_position = copy(candidate)
                    end
                end
            end
        end

        push!(best_history, best_value)
        push!(mean_history, sum(values) / length(values))
        push!(positions_history, snapshot_positions(positions))
    end

    AlgorithmResult(copy(best_position), best_value, best_history, mean_history, positions_history, params.iterations)
end

function timed_run(f::Function)
    started = time_ns()
    result = f()
    elapsed = (time_ns() - started) / 1e9
    (result = result, elapsed = elapsed)
end

function render_comparison_plot(
    output_path::AbstractString,
    function_name::AbstractString,
    pso_result::AlgorithmResult,
    fish_result::AlgorithmResult,
)
    _require_plots()
    Plots.gr()
    try
        Plots.default(fontfamily = "DejaVu Sans")
    catch
        nothing
    end

    mkpath(dirname(output_path))
    x = collect(0:length(pso_result.best_history) - 1)
    plt = Plots.plot(
        x,
        pso_result.best_history;
        linewidth = 2,
        color = :firebrick,
        label = "PSO",
        xlabel = "Итерация",
        ylabel = "Лучшее значение функции",
        title = "Сравнение на функции $(function_name)",
        legend = :topright,
    )
    Plots.plot!(
        plt,
        x,
        fish_result.best_history;
        linewidth = 2,
        color = :royalblue,
        label = "Косяк рыб",
    )
    redirect_stderr(devnull) do
        Plots.savefig(plt, output_path)
    end
    output_path
end

function sampled_frame_indices(frame_count::Int; max_frames::Int = 80)
    if frame_count <= max_frames
        return collect(1:frame_count)
    end
    indices = round.(Int, range(1, frame_count, length = max_frames))
    unique(indices)
end

function render_search_gif(
    output_path::AbstractString,
    function_spec::FunctionSpec,
    method_label::AbstractString,
    result::AlgorithmResult;
    fps::Int = 10,
)
    _require_plots()
    Plots.gr()
    try
        Plots.default(fontfamily = "DejaVu Sans")
    catch
        nothing
    end

    mkpath(dirname(output_path))

    xs = range(function_spec.lower[1], function_spec.upper[1], length = 220)
    ys = range(function_spec.lower[2], function_spec.upper[2], length = 220)
    z = [function_spec.objective([x, y]) for y in ys, x in xs]
    frame_indices = sampled_frame_indices(length(result.positions_history))

    anim = Plots.Animation()
    for frame_index in frame_indices
        positions = result.positions_history[frame_index]
        best_index = min(frame_index, length(result.best_history))
        plt = Plots.contour(
            xs,
            ys,
            z;
            levels = 30,
            color = :turbo,
            linewidth = 1.0,
            fill = false,
            aspect_ratio = :equal,
            xlabel = "x1",
            ylabel = "x2",
            title = "$(function_spec.name): $(method_label), итерация $(frame_index - 1)",
            legend = :topright,
            size = (1000, 700),
        )
        Plots.scatter!(
            plt,
            positions[:, 1],
            positions[:, 2];
            color = :black,
            alpha = 0.8,
            markersize = 4,
            label = "Текущие позиции",
        )
        Plots.scatter!(
            plt,
            [result.best_position[1]],
            [result.best_position[2]];
            color = :red,
            marker = :star5,
            markersize = 10,
            label = "Лучшее найденное",
        )
        Plots.annotate!(
            plt,
            function_spec.lower[1] + 0.03 * (function_spec.upper[1] - function_spec.lower[1]),
            function_spec.upper[2] - 0.06 * (function_spec.upper[2] - function_spec.lower[2]),
            Plots.text(@sprintf("best = %.6f", result.best_history[best_index]), 10, :black),
        )
        Plots.frame(anim, plt)
    end

    redirect_stderr(devnull) do
        Plots.gif(anim, output_path; fps = fps)
    end
    output_path
end

function run_comparison(function_spec::FunctionSpec, output_dir::AbstractString)
    pso_run = timed_run(() ->
        particle_swarm_optimization(
            function_spec.objective,
            function_spec.lower,
            function_spec.upper;
            params = function_spec.pso_params,
        )
    )
    fish_run = timed_run(() ->
        fish_school_search(
            function_spec.objective,
            function_spec.lower,
            function_spec.upper;
            params = function_spec.fish_params,
        )
    )

    render_comparison_plot(
        joinpath(output_dir, "$(function_spec.slug)_comparison.png"),
        function_spec.name,
        pso_run.result,
        fish_run.result,
    )
    render_search_gif(
        joinpath(output_dir, "$(function_spec.slug)_pso.gif"),
        function_spec,
        "PSO",
        pso_run.result,
    )
    render_search_gif(
        joinpath(output_dir, "$(function_spec.slug)_fish.gif"),
        function_spec,
        "Косяк рыб",
        fish_run.result,
    )

    println(function_spec.name)
    @printf("  PSO: %.6f (%.3f c)\n", pso_run.result.best_value, pso_run.elapsed)
    @printf("    точка: [%s]\n", join(map(x -> @sprintf("%.6f", x), pso_run.result.best_position), ", "))
    @printf("  Косяк рыб: %.6f (%.3f c)\n", fish_run.result.best_value, fish_run.elapsed)
    @printf("    точка: [%s]\n", join(map(x -> @sprintf("%.6f", x), fish_run.result.best_position), ", "))
    println()
end

function clear_output_dir(output_dir::AbstractString)
    if isdir(output_dir)
        for entry in readdir(output_dir; join = true)
            rm(entry; force = true, recursive = true)
        end
    else
        mkpath(output_dir)
    end
end

function main()
    output_dir = joinpath(@__DIR__, "output")
    clear_output_dir(output_dir)

    functions = [
        FunctionSpec(
            "Функция Швефеля",
            "schwefel",
            schwefel,
            [-500.0, -500.0],
            [500.0, 500.0],
            ParticleSwarmParams(80, 300, 0.35, 0.35, 0.03),
            FishSchoolParams(60, 300, 0.12, 0.01, 0.06, 0.005, 1.0, 5.0),
        ),
        FunctionSpec(
            "Функция Растригина",
            "rastrigin",
            rastrigin,
            [-5.12, -5.12],
            [5.12, 5.12],
            ParticleSwarmParams(80, 300, 0.35, 0.35, 0.03),
            FishSchoolParams(60, 300, 0.12, 0.01, 0.06, 0.005, 1.0, 5.0),
        ),
        FunctionSpec(
            "Функция Розенброка",
            "rosenbrock",
            rosenbrock,
            [-2.0, -1.0],
            [2.0, 3.0],
            ParticleSwarmParams(60, 350, 0.35, 0.35, 0.03),
            FishSchoolParams(50, 350, 0.15, 0.01, 0.08, 0.005, 1.0, 5.0),
        ),
    ]

    for function_spec in functions
        run_comparison(function_spec, output_dir)
    end
end

main()
