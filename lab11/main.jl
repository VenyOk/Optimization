using LinearAlgebra
# using Plots
using ForwardDiff

# function rosenbrock(x)
#     n = length(x)
#     sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:n-1)
# end
#
# function rosenbrock_grad(x)
#     n = length(x)
#     g = zeros(n)
#     for i in 1:n-1
#         g[i] += -400 * x[i] * (x[i+1] - x[i]^2) - 2 * (1 - x[i])
#         g[i+1] += 200 * (x[i+1] - x[i]^2)
#     end
#     g
# end
#
# function schwefel(x)
#     418.9829 * length(x) - sum(x[i] * sin(sqrt(abs(x[i]))) for i in eachindex(x))
# end
#
# function schwefel_grad(x)
#     n = length(x)
#     g = zeros(n)
#     for i in 1:n
#         xi = x[i]
#         if abs(xi) > 1e-10
#             g[i] = -sin(sqrt(abs(xi))) - xi * cos(sqrt(abs(xi))) * (0.5 / sqrt(abs(xi))) * sign(xi)
#         end
#     end
#     g
# end
#
# function rastrigin(x)
#     10 * length(x) + sum(x[i]^2 - 10 * cos(2 * π * x[i]) for i in eachindex(x))
# end
#
# function rastrigin_grad(x)
#     2 * x .+ 20 * π * sin.(2 * π * x)
# end
#
# function dfp(f, grad_f, x0; tol=1e-8, max_iter=1000)
#     n = length(x0)
#     x = copy(x0)
#     η = Matrix{Float64}(I, n, n)
#     path = [copy(x)]
#     g = grad_f(x)
#     for k in 1:max_iter
#         if norm(g) < tol
#             break
#         end
#         d = -η * g
#         Δx = d
#         x = x + Δx
#         push!(path, copy(x))
#         g_new = grad_f(x)
#         Δg = g_new - g
#         g = g_new
#         dot_Δx_Δg = dot(Δx, Δg)
#         if dot_Δx_Δg > 1e-14
#             z = η * Δg
#             η = η + (Δx * Δx') / dot_Δx_Δg - (z * z') / dot(Δg, z)
#         end
#     end
#     n_iter = length(path) - 1
#     x, path, n_iter
# end

function dfp_with_AB(grad_f, x0; tol=1e-8, max_iter=1000, exact_n_iter=nothing, step_α=nothing)
    n = length(x0)
    x = copy(x0)
    η = Matrix{Float64}(I, n, n)
    sum_A = zeros(n, n)
    sum_B = zeros(n, n)
    g = grad_f(x)
    n_iter = 0
    for k in 1:max_iter
        n_iter = k
        if exact_n_iter === nothing && norm(g) < tol
            break
        end
        if exact_n_iter !== nothing && k > exact_n_iter
            break
        end
        d = -η * g
        α = step_α !== nothing ? step_α(x, d, g) : 1.0
        Δx = α * d
        x = x + Δx
        g_new = grad_f(x)
        Δg = g_new - g
        g = g_new
        dot_Δx_Δg = dot(Δx, Δg)
        if dot_Δx_Δg > 1e-14
            z = η * Δg
            A_k = (Δx * Δx') / dot_Δx_Δg
            B_k = (z * z') / dot(Δg, z)
            sum_A += A_k
            sum_B += B_k
            η = η + A_k - B_k
        end
    end
    x, sum_A, sum_B, n_iter
end

# function plot_trajectory_2d(f, x_min, x_max, path, title_str; resolution=200)
#     xs = range(x_min[1], x_max[1], length=resolution)
#     ys = range(x_min[2], x_max[2], length=resolution)
#     Z = [f([x, y]) for x in xs, y in ys]
#     p = contour(xs, ys, Z', levels=30, color=:viridis, fill=true)
#     plot!(p, [pt[1] for pt in path], [pt[2] for pt in path], color=:red, lw=2, label="")
#     scatter!(p, [path[1][1]], [path[1][2]], color=:green, markersize=8, label="старт")
#     scatter!(p, [path[end][1]], [path[end][2]], color=:blue, markersize=8, label="конец")
#     xlims!(p, (x_min[1], x_max[1]))
#     ylims!(p, (x_min[2], x_max[2]))
#     title!(p, title_str)
#     xlabel!(p, "x₁")
#     ylabel!(p, "x₂")
#     p
# end
#
# x0_rosen = [-1.5, 2.0]
# x_opt_rosen, path_rosen, iter_rosen = dfp(rosenbrock, rosenbrock_grad, x0_rosen)
# p1 = plot_trajectory_2d(rosenbrock, [-2.0, -1.0], [2.5, 3.0], path_rosen, "Розенброк")
# savefig(p1, "rosenbrock_dfp.png")
#
# x0_schwefel = [400.0, 400.0]
# x_opt_schwefel, path_schwefel, iter_schwefel = dfp(schwefel, schwefel_grad, x0_schwefel)
# p2 = plot_trajectory_2d(schwefel, [250, 250], [450, 450], path_schwefel, "Швефель")
# savefig(p2, "schwefel_dfp.png")
#
# x0_rastrigin = [1.0, 1.0]
# x_opt_rastrigin, path_rastrigin, iter_rastrigin = dfp(rastrigin, rastrigin_grad, x0_rastrigin)
# p3 = plot_trajectory_2d(rastrigin, [-5.0, -5.0], [5.0, 5.0], path_rastrigin, "Растригин")
# savefig(p3, "rastrigin_dfp.png")
#
# println("Розенброк:  x* = ", x_opt_rosen, "  f(x*) = ", rosenbrock(x_opt_rosen), "  итераций: ", iter_rosen)
# println("Швефель:   x* = ", x_opt_schwefel, "  f(x*) = ", schwefel(x_opt_schwefel), "  итераций: ", iter_schwefel)
# println("Растригин: x* = ", x_opt_rastrigin, "  f(x*) = ", rastrigin(x_opt_rastrigin), "  итераций: ", iter_rastrigin)

# f(x₁, x₂) = 2x₁² + x₁x₂ + 1.5x₂²  (гессиан Q = [4 1; 1 3])
f_quad(x) = 2.0 * x[1]^2 + x[1] * x[2] + 1.5 * x[2]^2
grad_quad(x) = [4.0 * x[1] + x[2], x[1] + 3.0 * x[2]]
Q = [4.0 1.0; 1.0 3.0]
x0_quad = [10.0, 10.0]
n_dim = 2
step_quad(x, d, g) = -dot(d, g) / dot(d, Q * d)
x_opt_quad, sum_A, sum_B, iter_quad = dfp_with_AB(grad_quad, x0_quad; exact_n_iter=n_dim, step_α=step_quad)
H_quad = ForwardDiff.hessian(f_quad, x_opt_quad)
H_inv = inv(H_quad)
E = Matrix{Float64}(I, 2, 2)
println("||ΣA - H^(-1)|| = ", norm(sum_A - H_inv))
println("||ΣB - E|| = ", norm(sum_B - E))
println("Значение H^(-1):")
display(H_inv)
println("Значение Σ A^(i):")
display(sum_A)
println("ЗначениеΣ B^(i):")
display(sum_B)
