import Pkg
Pkg.activate(@__DIR__; io=devnull)

using LinearAlgebra
using Plots

plotly()

const EPS = 1.0e-8

struct LPCase
    title::String
    c::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    plot_limits::Tuple{Float64, Float64}
    filename::String
end

function with_nonneg(A, b)
    Aall = [A; -1.0 0.0; 0.0 -1.0]
    ball = [b; 0.0; 0.0]
    Aall, ball
end

function is_feasible(A, b, p; tol=1.0e-7)
    all(A * p .<= b .+ tol)
end

function push_unique!(points, p; tol=1.0e-7)
    q = collect(p)
    for existing in points
        if norm(existing - q) <= tol
            return
        end
    end
    push!(points, q)
end

function intersections(A, b)
    points = Vector{Vector{Float64}}()
    m = size(A, 1)
    for i in 1:(m - 1)
        for j in (i + 1):m
            M = [A[i, 1] A[i, 2]; A[j, 1] A[j, 2]]
            if abs(det(M)) <= EPS
                continue
            end
            p = M \ [b[i], b[j]]
            if is_feasible(A, b, p)
                push_unique!(points, p)
            end
        end
    end
    points
end

function sort_polygon(points)
    if length(points) <= 1
        return points
    end
    center = zeros(2)
    for p in points
        center .+= p
    end
    center ./= length(points)
    sort(points, by=p -> atan(p[2] - center[2], p[1] - center[1]))
end

function clipped_polygon(A, b, xmax, ymax)
    Aclip = [A; 1.0 0.0; 0.0 1.0]
    bclip = [b; xmax; ymax]
    sort_polygon(intersections(Aclip, bclip))
end

function feasible_rays(A)
    rays = Vector{Vector{Float64}}()
    for i in 1:size(A, 1)
        a = A[i, :]
        for d in ([a[2], -a[1]], [-a[2], a[1]])
            v = collect(d)
            if norm(v) <= EPS
                continue
            end
            if all(A * v .<= 1.0e-7)
                v ./= norm(v)
                push_unique!(rays, v; tol=1.0e-6)
            end
        end
    end
    rays
end

function unbounded_descent(A, c)
    rays = feasible_rays(A)
    if isempty(rays)
        return false, zeros(2)
    end
    values = [dot(c, r) for r in rays]
    k = argmin(values)
    values[k] < -1.0e-7, rays[k]
end

function minimum_set(A, b, c, vertices)
    values = [dot(c, v) for v in vertices]
    minval = minimum(values)
    points = Vector{Vector{Float64}}()
    for i in 1:size(A, 1)
        M = [A[i, 1] A[i, 2]; c[1] c[2]]
        if abs(det(M)) <= EPS
            continue
        end
        p = M \ [b[i], minval]
        if is_feasible(A, b, p) && abs(dot(c, p) - minval) <= 1.0e-6
            push_unique!(points, p)
        end
    end
    for p in vertices
        if abs(dot(c, p) - minval) <= 1.0e-6
            push_unique!(points, p)
        end
    end
    if isempty(points)
        return minval, :point, [vertices[argmin(values)]]
    end
    tangent = norm(c) <= EPS ? [1.0, 0.0] : [-c[2], c[1]]
    scores = [dot(tangent, p) for p in points]
    i1 = argmin(scores)
    i2 = argmax(scores)
    if scores[i2] - scores[i1] <= 1.0e-6
        return minval, :point, [points[i1]]
    end
    minval, :segment, [points[i1], points[i2]]
end

function line_box_points(n, alpha, xmax, ymax)
    points = Vector{Vector{Float64}}()
    if abs(n[2]) > EPS
        for x in (0.0, xmax)
            y = (alpha - n[1] * x) / n[2]
            if -1.0e-7 <= y <= ymax + 1.0e-7
                push_unique!(points, [clamp(x, 0.0, xmax), clamp(y, 0.0, ymax)])
            end
        end
    end
    if abs(n[1]) > EPS
        for y in (0.0, ymax)
            x = (alpha - n[2] * y) / n[1]
            if -1.0e-7 <= x <= xmax + 1.0e-7
                push_unique!(points, [clamp(x, 0.0, xmax), clamp(y, 0.0, ymax)])
            end
        end
    end
    if length(points) <= 2
        return points
    end
    tangent = [-n[2], n[1]]
    scores = [dot(tangent, p) for p in points]
    [points[argmin(scores)], points[argmax(scores)]]
end

function ray_box_endpoint(start, dir, xmax, ymax)
    ts = Float64[]
    if abs(dir[1]) > EPS
        push!(ts, ((dir[1] > 0 ? xmax : 0.0) - start[1]) / dir[1])
    end
    if abs(dir[2]) > EPS
        push!(ts, ((dir[2] > 0 ? ymax : 0.0) - start[2]) / dir[2])
    end
    ts = [t for t in ts if t > 0]
    if isempty(ts)
        return start
    end
    start .+ 0.95 * minimum(ts) .* dir
end

function polygon_center(points, xmax, ymax)
    if isempty(points)
        return [xmax / 2, ymax / 2]
    end
    center = zeros(2)
    for p in points
        center .+= p
    end
    center ./= length(points)
end

function objective_grid(c, xs, ys, A, b; feasible_only=false)
    [feasible_only && !is_feasible(A, b, [x, y]) ? NaN : c[1] * x + c[2] * y for y in ys, x in xs]
end

clean_value(x) = abs(x) <= 1.0e-9 ? 0.0 : x

function draw_case(case::LPCase)
    A, b = with_nonneg(case.A, case.b)
    vertices = intersections(A, b)
    if isempty(vertices)
        println("$(case.title): допустимая область пуста")
        return
    end

    xmax, ymax = case.plot_limits
    region = clipped_polygon(A, b, xmax, ymax)
    unbounded, ray = unbounded_descent(A, case.c)
    minval = NaN
    kind = :none
    opt = Vector{Vector{Float64}}()

    if !unbounded
        minval, kind, opt = minimum_set(A, b, case.c, vertices)
    end

    p2 = plot(
        title=case.title * " | 2D",
        xlabel="x1",
        ylabel="x2",
        xlim=(0, xmax),
        ylim=(0, ymax),
        aspect_ratio=:equal,
        legend=:topright,
        grid=true,
        framestyle=:zerolines
    )

    if length(region) >= 3
        shape = Shape([p[1] for p in region], [p[2] for p in region])
        plot!(p2, shape, c=:lightblue, fillalpha=0.35, linealpha=0, label="Допустимая область")
    end

    for i in 1:size(A, 1)
        pts = line_box_points(A[i, :], b[i], xmax, ymax)
        if length(pts) == 2
            plot!(p2, [pts[1][1], pts[2][1]], [pts[1][2], pts[2][2]], color=:black, linewidth=1.5, label="")
        end
    end

    ref_points = isempty(region) ? vertices : region
    level_values = [dot(case.c, p) for p in ref_points]
    lo = minimum(level_values)
    hi = maximum(level_values)
    if abs(hi - lo) <= EPS
        hi = lo + 1.0
    end
    levels = collect(range(lo, hi, length=6))

    for (idx, level) in enumerate(levels)
        pts = line_box_points(case.c, level, xmax, ymax)
        if length(pts) == 2
            label = idx == 1 ? "Линии уровня" : ""
            plot!(p2, [pts[1][1], pts[2][1]], [pts[1][2], pts[2][2]], color=:gray, linestyle=:dash, linewidth=1.0, alpha=0.7, label=label)
        end
    end

    if unbounded
        seed = is_feasible(A, b, [0.0, 0.0]) ? [0.0, 0.0] : vertices[argmin([norm(v) for v in vertices])]
        ray_end = ray_box_endpoint(seed, ray, xmax, ymax)
        plot!(p2, [seed[1], ray_end[1]], [seed[2], ray_end[2]], color=:red, linewidth=4, linestyle=:dash, label="Уход к -∞")
        println("$(case.title): минимум не найден, убывание уходит в бесконечность")
    elseif kind == :point
        p = opt[1]
        scatter!(p2, [p[1]], [p[2]], color=:red, markersize=8, label="Точка минимума")
        opt_line = line_box_points(case.c, minval, xmax, ymax)
        if length(opt_line) == 2
            plot!(p2, [opt_line[1][1], opt_line[2][1]], [opt_line[1][2], opt_line[2][2]], color=:red, linewidth=2.5, label="")
        end
        println("$(case.title): точка минимума = ($(round(clean_value(p[1]), digits=4)), $(round(clean_value(p[2]), digits=4))), f = $(round(clean_value(minval), digits=4))")
    else
        p1, p2s = opt
        plot!(p2, [p1[1], p2s[1]], [p1[2], p2s[2]], color=:red, linewidth=5, label="Отрезок минимума")
        scatter!(p2, [p1[1], p2s[1]], [p1[2], p2s[2]], color=:red, markersize=6, label="")
        println("$(case.title): отрезок минимума = [($(round(clean_value(p1[1]), digits=4)), $(round(clean_value(p1[2]), digits=4))) ; ($(round(clean_value(p2s[1]), digits=4)), $(round(clean_value(p2s[2]), digits=4)))], f = $(round(clean_value(minval), digits=4))")
    end

    xs = range(0, xmax, length=60)
    ys = range(0, ymax, length=60)
    z_plane = objective_grid(case.c, xs, ys, A, b; feasible_only=false)
    z_feasible = objective_grid(case.c, xs, ys, A, b; feasible_only=true)

    p3 = plot(
        title=case.title * " | 3D",
        xlabel="x1",
        ylabel="x2",
        zlabel="f(x)",
        legend=:topright,
        colorbar=false
    )

    surface!(p3, xs, ys, z_plane, color=:gray, alpha=0.2, label="Плоскость f(x1, x2)")
    surface!(p3, xs, ys, z_feasible, color=:viridis, alpha=0.8, label="Допустимая часть")

    if unbounded
        seed = is_feasible(A, b, [0.0, 0.0]) ? [0.0, 0.0] : vertices[argmin([norm(v) for v in vertices])]
        ray_end = ray_box_endpoint(seed, ray, xmax, ymax)
        plot!(p3, [seed[1], ray_end[1]], [seed[2], ray_end[2]], [dot(case.c, seed), dot(case.c, ray_end)], color=:red, linewidth=4, linestyle=:dash, label="Уход к -∞")
    elseif kind == :point
        p = opt[1]
        scatter!(p3, [p[1]], [p[2]], [dot(case.c, p)], color=:red, markersize=6, label="Минимум")
    else
        p1, p2s = opt
        plot!(p3, [p1[1], p2s[1]], [p1[2], p2s[2]], [dot(case.c, p1), dot(case.c, p2s)], color=:red, linewidth=6, label="Отрезок минимума")
        scatter!(p3, [p1[1], p2s[1]], [p1[2], p2s[2]], [dot(case.c, p1), dot(case.c, p2s)], color=:red, markersize=5, label="")
    end

    fig = plot(p2, p3, layout=(1, 2), size=(1500, 650))
    outdir = joinpath(@__DIR__, "plots")
    mkpath(outdir)
    savefig(fig, joinpath(outdir, case.filename))
    savefig(fig, joinpath(outdir, replace(case.filename, ".png" => ".html")))
    try
        display(fig)
    catch
    end
end

default(size=(1500, 650))

cases = [
    LPCase(
        "Случай 1: точка минимума",
        [-2.0, 1.0],
        [1.0 1.0; 1.0 0.0; 0.0 1.0],
        [8.0, 6.0, 5.0],
        (8.0, 6.0),
        "case_1_point.png"
    ),
    LPCase(
        "Случай 2: отрезок минимума",
        [-1.0, -1.0],
        [1.0 1.0; 1.0 0.0; 0.0 1.0],
        [4.0, 4.0, 4.0],
        (5.0, 5.0),
        "case_2_segment.png"
    ),
    LPCase(
        "Случай 3: минимум на бесконечности",
        [-1.0, -1.0],
        [-1.0 1.0; 1.0 -1.0; 1.0 -2.0],
        [2.0, 5.0, 8.0],
        (12.0, 12.0),
        "case_3_unbounded.png"
    )
]

for case in cases
    draw_case(case)
end
