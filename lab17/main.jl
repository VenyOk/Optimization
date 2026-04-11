using Random
using Printf
using Plots

f(x) = 5 - 24x + 17x^2 - (11 / 3) * x^3 + (1 / 4) * x^4

function to_x(ch::BitVector, a::Float64, b::Float64, h::Float64)
    v = 0
    for bit in ch
        v = (v << 1) | Int(bit)
    end
    m = (1 << length(ch)) - 1
    x = a + (b - a) * v / m
    a + round((x - a) / h) * h
end

function pop_x(pop::Vector{BitVector}, a::Float64, b::Float64, h::Float64)
    [to_x(ch, a, b, h) for ch in pop]
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

function ga(; a = 0.0, b = 7.0, h = 1.0, bits = 3, n = 8, iters = 25, pc = 0.8, pm = 0.3, seed = 42)
    rng = MersenneTwister(seed)
    pop = [bitrand(rng, bits) for _ in 1:n]
    hist = Vector{Vector{Float64}}()
    mut_hist = Vector{Vector{Float64}}()
    push!(hist, pop_x(pop, a, b, h))
    push!(mut_hist, Float64[])

    x = pop_x(pop, a, b, h)
    y = f.(x)
    k = argmin(y)
    best_x = x[k]
    best_y = y[k]

    for _ in 1:iters
        x = pop_x(pop, a, b, h)
        y = f.(x)
        k = argmin(y)
        if y[k] < best_y
            best_y = y[k]
            best_x = x[k]
        end

        kids = BitVector[]
        mut_x = Float64[]
        while length(kids) < n
            i = pick(y, rng)
            j = pick(y, rng)
            c1, c2 = cross(pop[i], pop[j], pc, rng)
            m1 = mut!(c1, pm, rng)
            m2 = mut!(c2, pm, rng)
            push!(kids, c1)
            if m1
                push!(mut_x, to_x(c1, a, b, h))
            end
            if length(kids) < n
                push!(kids, c2)
                if m2
                    push!(mut_x, to_x(c2, a, b, h))
                end
            end
        end

        all = vcat(pop, kids)
        all_x = pop_x(all, a, b, h)
        all_y = f.(all_x)
        ord = sortperm(all_y)
        pop = [copy(all[ord[i]]) for i in 1:n]
        push!(hist, pop_x(pop, a, b, h))
        push!(mut_hist, mut_x)
    end

    x = pop_x(pop, a, b, h)
    y = f.(x)
    k = argmin(y)
    if y[k] < best_y
        best_y = y[k]
        best_x = x[k]
    end

    conv = -1
    for i in eachindex(hist)
        if all(abs.(hist[i] .- best_x) .< 1e-9)
            conv = i - 1
            break
        end
    end

    best_x, best_y, conv, hist, mut_hist
end

function make_gif(hist::Vector{Vector{Float64}}, mut_hist::Vector{Vector{Float64}}, file::String; a = 0.0, b = 7.0)
    xx = range(a, b, length = 400)
    yy = f.(xx)
    n = length(hist)

    anim = @animate for i in 1:n
        cur_x = hist[i]
        cur_y = f.(cur_x)
        p = plot(xx, yy, color = :black, linewidth = 2, label = "f(x)", xlabel = "x", ylabel = "f(x)", title = "Итерация $(i - 1) / $(n - 1)", legend = :topright)
        if i == 1
            scatter!(p, cur_x, cur_y, color = :blue, markerstrokecolor = :blue, markersize = 7, label = "Initial")
        else
            scatter!(p, cur_x, cur_y, color = :red, markerstrokecolor = :red, markersize = 7, label = "Промежуточные")
        end
        if !isempty(mut_hist[i])
            scatter!(p, mut_hist[i], f.(mut_hist[i]), color = :black, markerstrokecolor = :black, markersize = 8, label = "Мутировавшие")
        end
    end

    gif(anim, file, fps = 4)
end

a = 0.0
b = 7.0

best_x1, best_y1, conv1, hist1, mut_hist1 = ga(a = a, b = b, h = 1.0, bits = 3, n = 8, iters = 50, pc = 0.8, pm = 0.3, seed = 67)
file1 = joinpath(@__DIR__, "ga_step_1.gif")
make_gif(hist1, mut_hist1, file1, a = a, b = b)

best_x2, best_y2, conv2, hist2, mut_hist2 = ga(a = a, b = b, h = 0.01, bits = 10, n = 80, iters = 50, pc = 0.8, pm = 0.01, seed = 67)
file2 = joinpath(@__DIR__, "ga_step_001.gif")
make_gif(hist2, mut_hist2, file2, a = a, b = b)

@printf("step = 1, x = %.2f, f(x) = %.6f, итерации = %d\n", best_x1, best_y1, conv1)
@printf("step = 0.01, x = %.2f, f(x) = %.6f, итерации = %d\n", best_x2, best_y2, conv2)
