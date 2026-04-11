using LinearAlgebra
using Random
using Plots

to_bits(x) = reverse(digits(x, base = 2, pad = 3))
from_bits(b) = b[1] * 4 + b[2] * 2 + b[3]
clamp_gene(x) = clamp(x, 0, 6)

function gene_crossing(a, b)
    ba = to_bits(a)
    bb = to_bits(b)
    p = rand(1:2)
    c1 = vcat(ba[1:p], bb[p + 1:end])
    c2 = vcat(bb[1:p], ba[p + 1:end])
    return clamp_gene(from_bits(c1)), clamp_gene(from_bits(c2))
end

function mutate_gene(x)
    b = to_bits(x)
    j = rand(1:3)
    b[j] = 1 - b[j]
    return clamp_gene(from_bits(b))
end

function crossover_whole_bits(a::Vector{Int}, b::Vector{Int})
    ba = flatten_bits(a)
    bb = flatten_bits(b)
    p = rand(1:length(ba) - 1)
    c1_bits = vcat(ba[1:p], bb[p + 1:end])
    c2_bits = vcat(bb[1:p], ba[p + 1:end])
    return bits_to_vec(c1_bits), bits_to_vec(c2_bits)
end

function crossover_by_gene(a::Vector{Int}, b::Vector{Int})
    n = length(a)
    c1 = Vector{Int}(undef, n)
    c2 = Vector{Int}(undef, n)

    for i in 1:n
        c1[i], c2[i] = gene_crossing(a[i], b[i])
    end

    return c1, c2
end

function mutate_image_once!(child::Vector{Int})
    k = rand(1:length(child))
    child[k] = mutate_gene(child[k])
end

function draw_letter!(img::Matrix{Int}, letter::Symbol)
    n = size(img, 1)
    mid = cld(n, 2)

    if letter == :X
        for i in 1:n
            img[i, i] = 0
            img[i, n - i + 1] = 0
        end
    elseif letter == :H
        img[:, 1] .= 0
        img[:, n] .= 0
        img[mid, :] .= 0
    elseif letter == :T
        img[1, :] .= 0
        img[:, mid] .= 0
    elseif letter == :L
        img[:, 1] .= 0
        img[n, :] .= 0
    elseif letter == :O
        img[1, :] .= 0
        img[n, :] .= 0
        img[:, 1] .= 0
        img[:, n] .= 0
    elseif letter == :A
        img[1, :] .= 0
        img[:, 1] .= 0
        img[:, n] .= 0
        img[mid, :] .= 0
    else
        error("Неизвестная буква: $letter. Используйте :X, :H, :T, :L, :O или :A")
    end
end

function make_target_image(n::Int, letter::Symbol = :A)
    img = Matrix{Int}(undef, n, n)

    for i in 1:n, j in 1:n
        img[i, j] = mod(i + j, 6) + 1
    end

    draw_letter!(img, letter)

    return img
end

img_to_vec(img) = vec(copy(img))
vec_to_img(v, n) = reshape(copy(v), n, n)

function flatten_bits(v)
    bits = Int[]
    sizehint!(bits, 3 * length(v))
    for x in v
        append!(bits, to_bits(x))
    end
    return bits
end

function bits_to_vec(bits)
    m = div(length(bits), 3)
    out = Vector{Int}(undef, m)
    for i in 1:m
        out[i] = clamp_gene(from_bits(bits[3i - 2:3i]))
    end
    return out
end

function bool_dist(a, b)
    ba = flatten_bits(a)
    bb = flatten_bits(b)
    return sum(ba .!= bb)
end

function dist(ind, target, typ::Symbol)
    if typ == :bool
        return bool_dist(ind, target)
    else
        return euclid(ind, target)
    end
end

euclid(a, b) = norm(a .- b)

Random.seed!(14)

function genetic_image(;
    n::Int = 5,
    letter::Symbol = :A,
    variant::Symbol = :whole_bits,
    typ::Symbol = :bool,
    popsize::Int = 20,
    max_iter::Int = 200,
    pmut::Float64 = 0.3,
    eps::Float64 = 0.0,
    seed::Int = 14,
)
    Random.seed!(seed)

    target_img = make_target_image(n, letter)
    target = img_to_vec(target_img)

    population = [rand(0:6, n * n) for _ in 1:popsize]
    initial_population = [copy(ind) for ind in population]

    history = NamedTuple[]

    best_ind = nothing
    best_fit = Inf
    stop_iter = max_iter

    for iter in 1:max_iter
        parents = [copy(ind) for ind in population]
        children = Vector{Vector{Int}}()

        while length(children) < popsize
            pair = randperm(popsize)[1:2]
            p1 = parents[pair[1]]
            p2 = parents[pair[2]]

            if variant == :whole_bits
                c1, c2 = crossover_whole_bits(p1, p2)
            elseif variant == :by_gene
                c1, c2 = crossover_by_gene(p1, p2)
            else
                error("Неизвестный вариант: $variant")
            end

            push!(children, c1)
            length(children) < popsize && push!(children, c2)
        end

        children_before_mut = [copy(ch) for ch in children]

        for ch in children
            if rand() < pmut
                mutate_image_once!(ch)
            end
        end

        all_inds = vcat(parents, children)
        fits = [dist(ind, target, typ) for ind in all_inds]
        best_idx = sortperm(fits)[1:popsize]
        population = [copy(all_inds[i]) for i in best_idx]

        current_best = copy(population[1])
        current_fit = dist(current_best, target, typ)

        if current_fit < best_fit
            best_fit = current_fit
            best_ind = copy(current_best)
        end

        push!(
            history,
            (
                iter = iter,
                parents = parents,
                children_before = children_before_mut,
                children_after = [copy(ch) for ch in children],
                best = copy(current_best),
                best_fit = current_fit,
            ),
        )

        if current_fit <= eps
            stop_iter = iter
            break
        end
    end

    return (
        n = n,
        letter = letter,
        target_img = target_img,
        target_vec = target,
        initial_population = initial_population,
        final_population = population,
        best = best_ind,
        best_fit = best_fit,
        history = history,
        stop_iter = stop_iter,
        variant = variant,
        typ = typ,
    )
end

plotlyjs()

function image_plot(img::Matrix{Int}, ttl::String)
    heatmap(
        img[end:-1:1, :],
        c = cgrad(:grays),
        clims = (0, 6),
        colorbar = false,
        aspect_ratio = 1,
        framestyle = :box,
        xticks = false,
        yticks = false,
        title = ttl,
    )
end

function make_gif(res; filename = nothing, fps = 5)
    n = res.n
    target_img = res.target_img

    if filename === nothing
        filename = "ga_image_$(n)x$(n)_$(res.letter)_$(res.variant)_$(res.typ).gif"
    end

    anim = @animate for k in 1:length(res.history)
        best_img = vec_to_img(res.history[k].best, n)

        p1 = image_plot(
            best_img,
            "Текущее лучшее\nитерация=$(res.history[k].iter)",
        )

        p2 = image_plot(target_img, "Эталон")

        plot(p1, p2, layout = (1, 2), size = (900, 420))
    end

    return gif(anim, filename, fps = fps)
end

function run_one(n; letter = :A, variant, typ, eps, popsize = 20, max_iter = 300, pmut = 0.3, seed = 14)
    res = genetic_image(
        n = n,
        letter = letter,
        variant = variant,
        typ = typ,
        popsize = popsize,
        max_iter = max_iter,
        pmut = pmut,
        eps = eps,
        seed = seed,
    )

    println()
    println("$(n)x$(n)")
    println("Буква: $(letter)")
    println("Вариант: $(variant)")
    println("Расстояние: $(typ)")
    println("Лучшая ошибка: $(res.best_fit)")
    println("Итерация: $(res.stop_iter)")
    println()

    g = make_gif(res)
    display(g)

    return res
end

function run_all(; letter = :A, popsize = 30, max_iter = 400, pmut = 0.3, fps = 2)
    sizes = [3, 5, 9, 21]
    fpss = [5, 5, 10, 20]
    results = []

    println("Разрез по всему вектору и булево расстояние")
    for i in 1:length(sizes)
        eps = 0.0
        res = genetic_image(
            n = sizes[i],
            letter = letter,
            variant = :whole_bits,
            typ = :bool,
            popsize = popsize,
            max_iter = max_iter,
            pmut = pmut,
            eps = eps,
            seed = 14,
        )

        gif_name = "ga_$(sizes[i])x$(sizes[i])_$(letter)_whole_bits_bool.gif"
        g = make_gif(res; filename = gif_name, fps = fpss[i])
        display(g)

        push!(results, res)
    end

    println("Разрез внутри чисел и евклидово расстояние")
    for i in 1:length(sizes)
        eps = 1e-9
        res = genetic_image(
            n = sizes[i],
            letter = letter,
            variant = :by_gene,
            typ = :euclid,
            popsize = popsize,
            max_iter = max_iter,
            pmut = pmut,
            eps = eps,
            seed = 14,
        )

        gif_name = "ga_$(sizes[i])x$(sizes[i])_$(letter)_by_gene_euclid.gif"
        g = make_gif(res; filename = gif_name, fps = fpss[i])
        display(g)

        push!(results, res)
    end

    return results
end

run_one(
    5;
    letter = :A,
    variant = :whole_bits,
    typ = :bool,
    eps = 0.0,
    popsize = 30,
    max_iter = 300,
    pmut = 0.35,
    seed = 14,
)

println()

run_all(letter = :A, popsize = 30, max_iter = 500, pmut = 0.35, fps = 5)
