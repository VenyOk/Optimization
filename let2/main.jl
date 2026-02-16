using LinearAlgebra
using Plots
Plots.default(background=:white, foreground_color=:darkgray)

function initial_simplex1(n)
    P = (1 / (n*sqrt(2))) * (sqrt(n + 1) + n - 1)
    Q = (1 / (n*sqrt(2))) * (sqrt(n + 1) - 1)
    X = zeros(n, n + 1)
    for k in 2:n+1
        for j in 1:n
            X[j, k] = (j == k - 1) ? P : Q
        end
    end
    return X
end

function initial_simplex2(n)
    R = [sqrt(1 / (2 * i*(i + 1))) for i in 1:n]
    V = [sqrt(i / (2 * (i + 2))) for i in 1:n]
    X = zeros(n, n + 1)
    for j in 1:n
        X[j, 1] = -R[j]
    end
    for k in 2:n+1
        for j in 1:n
            X[j, k] = (j == k - 1) ? V[k-1] : -R[j]
        end
    end
    return X
end

function simple_simplex(f, x0; ε=1e-6, maxiter=10000, σ=0.5, history=nothing, init_method=1, ref_max_ratio=2.0)
    n = length(x0)
    S = (init_method == 1) ? initial_simplex1(n) : initial_simplex2(n)
    X = x0 .+ S
    fvals = [f(X[:, i]) for i in 1:n+1]
    iter = 0
    if history !== nothing
        push!(history, copy(X))
    end
    while iter < maxiter
        order = sortperm(fvals)
        fvals = fvals[order]
        X = X[:, order]
        x_centroid = sum(X[:, 1:n], dims=2) / n
        x_ref = 2 * x_centroid - X[:, n+1]
        d_worst = norm(X[:, n+1] - x_centroid)
        d_ref = norm(x_ref - x_centroid)
        if d_worst > 1e-12 && d_ref > ref_max_ratio * d_worst
            x_ref = x_centroid + (x_ref - x_centroid) .* (ref_max_ratio * d_worst / d_ref)
        end
        f_ref = f(x_ref)
        if f_ref < fvals[n+1]
            X[:, n+1] = x_ref
            fvals[n+1] = f_ref
        else
            for i in 2:n+1
                X[:, i] = X[:, 1] + σ * (X[:, i] - X[:, 1])
                fvals[i] = f(X[:, i])
            end
        end
        if history !== nothing
            push!(history, copy(X))
        end
        if sqrt(sum(sum((X[:, i] .- x_centroid).^2) for i in 1:n+1) / (n+1)) < ε
            break
        end
        iter += 1
    end
    return X[:, 1], fvals[1], iter
end

schwefel(x) = 418.9829 * length(x) - sum(x .* sin.(sqrt.(abs.(x))))
rastrigin(x) = 10 * length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))
rosenbrock(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:length(x)-1)

function plot_simplex_history(f, history, xlims, ylims, title_str, filename)
    all_x = Float64[]
    all_y = Float64[]
    for X in history
        append!(all_x, X[1, :])
        append!(all_y, X[2, :])
    end
    margin_x = max(0.1 * (maximum(all_x) - minimum(all_x)), 0.5)
    margin_y = max(0.1 * (maximum(all_y) - minimum(all_y)), 0.5)
    xmin = minimum(all_x) - margin_x
    xmax = maximum(all_x) + margin_x
    ymin = minimum(all_y) - margin_y
    ymax = maximum(all_y) + margin_y
    xmin = max(xmin, xlims[1])
    xmax = min(xmax, xlims[2])
    ymin = max(ymin, ylims[1])
    ymax = min(ymax, ylims[2])
    p = plot(background=:white, xlims=(xmin, xmax), ylims=(ymin, ymax))
    for (k, X) in enumerate(history)
        tri_x = [X[1, 1]; X[1, 2]; X[1, 3]; X[1, 1]]
        tri_y = [X[2, 1]; X[2, 2]; X[2, 3]; X[2, 1]]
        plot!(p, tri_x, tri_y, linewidth=2.5, color=:black, alpha=0.85, legend=false)
        if k == 1
            scatter!(p, X[1, :], X[2, :], markersize=4, color=:black, marker=:circle, legend=false)
        end
    end
    if length(history) >= 1
        X1 = history[1]
        annotate!(p, [(X1[1, j], X1[2, j], text(string(j), 8, :center)) for j in 1:3])
        scatter!(p, [X1[1, 1]], [X1[2, 1]], markersize=4, color=:blue, label="начало")
    end
    X = history[end]
    scatter!(p, [X[1, 1]], [X[2, 1]], markersize=4, color=:red, label="минимум")
    plot!(p, xlims=(xmin, xmax), ylims=(ymin, ymax), title=title_str, xlabel="x₁", ylabel="x₂")
    savefig(p, filename)
end

n = 2
x0_s = [400.0, 400.0]
x0_r = [3.0, 3.0]
x0_ros = [-1.5, 2.0]

println("Швеффель n=$n")
hist_s = []
x, f, it = simple_simplex(schwefel, x0_s, history=hist_s)
println("Точка: $x")
println("Значение функции: $f")
println("Итераций: $it")

println("\nРастригин n=$n")
hist_r = []
x, f, it = simple_simplex(rastrigin, x0_r, history=hist_r)
println("Точка: $x")
println("Значение функции: $f")
println("Итераций: $it")

println("\nРозенброк n=$n")
hist_ros = []
x, f, it = simple_simplex(rosenbrock, x0_ros, history=hist_ros)
println("Точка: $x")
println("Значение функции: $f")
println("Итераций: $it")

plot_simplex_history(schwefel, hist_s, (-500, 500), (-500, 500), "Швеффель", "let2_schwefel.png")
plot_simplex_history(rastrigin, hist_r, (-5, 5), (-5, 5), "Растригин", "let2_rastrigin.png")
plot_simplex_history(rosenbrock, hist_ros, (-2, 2.5), (-1, 2.5), "Розенброк", "let2_rosenbrock.png")
println("\nГрафики сохранены: let2_schwefel.png, let2_rastrigin.png, let2_rosenbrock.png")
