using LinearAlgebra

function f(x)
    return 9 * x[1]^2 + 2 * x[2]^2
end

function grad_f(x)
    return [18 * x[1], 4 * x[2]]
end


function exact_step_quadratic(g, p, A)
    denom = dot(p, A * p)
    α = (abs(denom) < 1e-14) ? 1.0 : -dot(g, p) / denom
    return max(α, 1e-10)
end

round3(x; digits=3) = round(x; digits=digits)
round3(v::AbstractArray; digits=3) = round.(v; digits=digits)

function bfgs(f, grad_f, x0; max_iter=1000, tol=1e-8, A=nothing, digits=3, round_each_step=true)
    n = length(x0)
    x = copy(x0)
    E = Matrix{Float64}(I, n, n)
    η = copy(E)
    ∇f = grad_f(x)
    if round_each_step
        x = round3(x; digits=digits)
        η = round3(η; digits=digits)
        ∇f = round3(∇f; digits=digits)
    end
    for k in 0:(max_iter - 1)
        println("=== Шаг k = $k ===")
        println("x^($k) = $(round3(x; digits=digits))")
        println("f(x^($k)) = $(round3(f(x); digits=digits))")
        println("∇f(x^($k)) = $(round3(∇f; digits=digits))")
        g_norm = norm(∇f)
        println("||∇f(x^($k))|| = $(round3(g_norm; digits=digits))")
        if g_norm < tol
            println("Критерий остановки выполнен. Решение: x = $(round3(x; digits=digits))")
            return x
        end
        print("η^($k) = ")
        println(round3(η; digits=digits))
        p = -η * ∇f
        if round_each_step
            p = round3(p; digits=digits)
        end
        println("p^($k) = -η·∇f = $(round3(p; digits=digits))")
        α = exact_step_quadratic(∇f, p, A)
        if round_each_step
            α = round3(α; digits=digits)
        end
        println("α^($k) = $(round3(α; digits=digits))")
        Δx = α * p
        if round_each_step
            Δx = round3(Δx; digits=digits)
        end
        println("Δx^($k) = α·p = $(round3(Δx; digits=digits))")
        x_new = x .+ Δx
        if round_each_step
            x_new = round3(x_new; digits=digits)
        end
        println("x^($(k+1)) = x^($k) + Δx^($k) = $(round3(x_new; digits=digits))")
        ∇f_new = grad_f(x_new)
        if round_each_step
            ∇f_new = round3(∇f_new; digits=digits)
        end
        println("∇f(x^($(k+1))) = $(round3(∇f_new; digits=digits))")
        Δg = ∇f_new .- ∇f
        if round_each_step
            Δg = round3(Δg; digits=digits)
        end
        println("Δg^($k) = ∇f(x^($(k+1))) - ∇f(x^($k)) = $(round3(Δg; digits=digits))")
        denom = dot(Δg, Δx)
        ρ = 1.0 / (denom + 1e-12)
        if round_each_step
            ρ = round3(ρ; digits=digits)
        end
        println("ρ^($k) = 1/(Δg·Δx) = $(round3(ρ; digits=digits))")
        V = E - ρ * (Δg * Δx')
        η = V' * η * V + ρ * (Δx * Δx')
        if round_each_step
            η = round3(η; digits=digits)
        end
        x = x_new
        ∇f = ∇f_new
    end
    println("Достигнуто максимальное число итераций. x = $(round3(x; digits=digits))")
    return x
end

x0 = [1.0, 1.0]
A_hess = [18.0 0.0; 0.0 4.0]
println()
x_opt = bfgs(f, grad_f, x0; A=A_hess, digits=3, round_each_step=true)
println()
println("Итоговое решение: $x_opt")
