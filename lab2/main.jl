using Plots

function svenn(f, x0, h)
    x1 = x0
    x2 = x0 + h
    if f(x2) > f(x1)
        h = -h
        x2 = x0 + h
    end
    while f(x2) < f(x1)
        x1 = x2
        x2 = x1 + h
        h = 2 * h
    end
    a = min(x1, x2)
    b = max(x1, x2)
    return a, b
end

function exhaustive_search(f, a, b, n)
    step = (b - a) / (n - 1)
    x_seq = Float64[]
    y_seq = Float64[]
    x_best = a
    f_best = f(a)
    push!(x_seq, a)
    push!(y_seq, f_best)
    for i = 2:n
        x = a + (i - 1) * step
        fx = f(x)
        push!(x_seq, x)
        push!(y_seq, fx)
        if fx < f_best
            f_best = fx
            x_best = x
        end
    end
    return x_best, x_seq, y_seq, n - 1
end

function dichotomy_search(f, a, b, eps)
    delta = eps / 4
    x_seq = Float64[]
    y_seq = Float64[]
    iters = 0
    while (b - a) / 2 > eps
        m = (a + b) / 2
        x1 = m - delta
        x2 = m + delta
        push!(x_seq, m)
        push!(y_seq, f(m))
        if f(x1) < f(x2)
            b = m
        else
            a = m
        end
        iters += 1
    end
    x_min = (a + b) / 2
    push!(x_seq, x_min)
    push!(y_seq, f(x_min))
    return x_min, x_seq, y_seq, iters
end

const PHI = (sqrt(5) - 1) / 2

function golden_section_search(f, a, b, eps)
    x_seq = Float64[]
    y_seq = Float64[]
    iters = 0
    x1 = b - (b - a) * PHI
    x2 = a + (b - a) * PHI
    f1 = f(x1)
    f2 = f(x2)
    while (b - a) > eps
        push!(x_seq, f1 < f2 ? x1 : x2)
        push!(y_seq, f1 < f2 ? f1 : f2)
        if f1 < f2
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) * PHI
            f1 = f(x1)
        else
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * PHI
            f2 = f(x2)
        end
        iters += 1
    end
    x_min = (a + b) / 2
    push!(x_seq, x_min)
    push!(y_seq, f(x_min))
    return x_min, x_seq, y_seq, iters
end

function fib(n)
    n <= 1 && return n
    a, b = 0, 1
    for _ in 2:n
        a, b = b, a + b
    end
    return b
end

function fibonacci_search(f, a, b, eps)
    n = 1
    while fib(n + 2) < (b - a) / eps
        n += 1
    end
    x_seq = Float64[]
    y_seq = Float64[]
    k = 1
    x1 = a + (b - a) * fib(n) / fib(n + 2)
    x2 = a + (b - a) * fib(n + 1) / fib(n + 2)
    f1 = f(x1)
    f2 = f(x2)
    while k < n
        push!(x_seq, f1 < f2 ? x1 : x2)
        push!(y_seq, f1 < f2 ? f1 : f2)
        if f1 > f2
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) * fib(n - k + 1) / fib(n - k + 2)
            f2 = f(x2)
        else
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) * fib(n - k) / fib(n - k + 2)
            f1 = f(x1)
        end
        k += 1
    end
    candidates = [a, b, (a + b) / 2, x1, x2]
    x_min = candidates[1]
    for c in candidates
        if f(c) < f(x_min)
            x_min = c
        end
    end
    push!(x_seq, x_min)
    push!(y_seq, f(x_min))
    return x_min, x_seq, y_seq, n
end

function num_derivative(f, x, delta = 1e-7)
    return (f(x + delta) - f(x - delta)) / (2 * delta)
end

function num_second_derivative(f, x, delta = 1e-6)
    return (f(x + delta) - 2 * f(x) + f(x - delta)) / (delta^2)
end

function prove_unimodality(f, a, b, n_points = 100)
    step = (b - a) / (n_points - 1)
    sign_changes = 0
    prev_der = num_derivative(f, a)
    for i in 2:n_points
        x = a + (i - 1) * step
        der = num_derivative(f, x)
        if (prev_der < 0 && der >= 0) || (prev_der > 0 && der <= 0)
            sign_changes += 1
        end
        prev_der = der
    end
    unimodal = (sign_changes == 1)
    println("Проверка унимодальности на [", a, ", ", b, "]: $unimodal")
    return unimodal
end

function prove_minimum_rule(f, x_min, precision = 1e-5)
    delta = max(1e-10, precision * 0.5)
    fpp = num_second_derivative(f, x_min, delta)
    f0 = f(x_min)
    rain_left = f(x_min - delta) >= f0 - 1e-12
    rain_right = f(x_min + delta) >= f0 - 1e-12
    is_min_rain = rain_left && rain_right
    println("Вторая производная в точке: ", fpp)
    println("Правило дождя: f(x*-h) >= f(x*), f(x*+h) >= f(x*) при h=", delta, " : ", rain_left, ", ", rain_right)
    println("Точка является минимумом: ", is_min_rain)
    return is_min_rain
end

function prove_maximum_rule(f, x_max, precision = 1e-5)
    delta = max(1e-10, precision * 0.5)
    fpp = num_second_derivative(f, x_max, delta)
    f0 = f(x_max)
    tol = 1e-9
    rain_left = f(x_max - delta) <= f0 + tol
    rain_right = f(x_max + delta) <= f0 + tol
    is_max_rain = rain_left && rain_right
    println("Вторая производная в точке: ", fpp)
    println("Правило дождя (макс): f(x*-h) <= f(x*), f(x*+h) <= f(x*) при h=", delta, " : ", rain_left, ", ", rain_right)
    println("Точка является максимумом: ", is_max_rain)
    return is_max_rain
end

f(x) = x > 0 ? -x * log(x) : 0.0
g(x) = -f(x)
eps = 1e-5

a, b = svenn(g, 0.5, 0.05)
println("Метод Свенна: интервал [", a, ", ", b, "]")

n_exh = max(100, min(Int(ceil((b - a) / eps)), 10000))
x_exh, xseq_exh, _, iter_exh = exhaustive_search(g, a, b, n_exh)
yseq_exh = f.(xseq_exh)
x_dich, xseq_dich, _, iter_dich = dichotomy_search(g, a, b, eps)
yseq_dich = f.(xseq_dich)
x_gold, xseq_gold, _, iter_gold = golden_section_search(g, a, b, eps)
yseq_gold = f.(xseq_gold)
x_fib, xseq_fib, _, iter_fib = fibonacci_search(g, a, b, eps)
yseq_fib = f.(xseq_fib)

prove_unimodality(f, a, b)

println("Максимум: x* = ", x_fib, ", f(x*) = ", f(x_fib))
println("Количество итераций:")
println("  Перебор: ", iter_exh)
println("  Дихотомия: ", iter_dich)
println("  Золотое сечение: ", iter_gold)
println("  Фибоначчи: ", iter_fib)

prove_maximum_rule(f, x_fib, eps)

x_plot = range(a, b, length = 500)
step_exh = max(1, length(xseq_exh) ÷ 50)
p = plot(x_plot, f.(x_plot), color = :gray, lw = 0.5, ls = :dot, label = "f(x)", legend = :topright, legendfontsize = 8)
plot!(p, xseq_exh[1:step_exh:end], yseq_exh[1:step_exh:end], color = :blue, lw = 0.8, ls = :solid, marker = :circle, ms = 2, label = "Перебор (n=$iter_exh)")
plot!(p, xseq_dich, yseq_dich, color = :red, lw = 1.2, ls = :dash, marker = :square, ms = 5, label = "Дихотомия (n=$iter_dich)")
plot!(p, xseq_gold, yseq_gold, color = :green, lw = 1.2, ls = :dashdot, marker = :diamond, ms = 5, label = "Золотое сечение (n=$iter_gold)")
plot!(p, xseq_fib, yseq_fib, color = :purple, lw = 1.2, ls = :dot, marker = :star5, ms = 6, label = "Фибоначчи (n=$iter_fib)")
vline!(p, [x_fib - eps, x_fib + eps], color = :black, lw = 0.8, ls = :dash, label = "Интервал точности ±ε")
hline!(p, [f(x_fib) - eps, f(x_fib) + eps], color = :black, lw = 0.8, ls = :dot, label = nothing)
xlims!(p, a - 0.02 * (b - a), b + 0.02 * (b - a))
xlabel!(p, "x")
ylabel!(p, "f(x)")
savefig(p, "lab2.png")
display(p)
