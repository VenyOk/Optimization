using Random
using Images
using ImageTransformations
using FileIO
using ColorTypes
using Statistics

function random_genotype()
    vcat(rand(-9:9, 15), rand(0:255, 3), rand(2:12))
end

function choose_active_genes(seed = nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    sort(randperm(rng, 15)[1:7])
end

function mutate_genotype(genes::Vector{Int}, active_idx::Vector{Int})
    child = copy(genes)
    allowed = vcat(active_idx, 16, 17, 18, 19)
    i = rand(allowed)
    if i <= 15
        delta = rand(Bool) ? 1 : -1
        child[i] = clamp(child[i] + delta, -9, 9)
    elseif i <= 18
        delta = rand([-10, 10])
        child[i] = clamp(child[i] + delta, 0, 255)
    else
        delta = rand(Bool) ? 1 : -1
        child[i] = clamp(child[i] + delta, 2, 12)
    end
    return child
end

function calculate_stems(genes, active_idx)
    g = genes[active_idx]
    return [
        (x = 0.0, y = 1.0 * g[1]),
        (x = 1.0 * g[2], y = 1.0 * g[3]),
        (x = 1.0 * g[4], y = 0.0),
        (x = 1.0 * g[5], y = -1.0 * g[6]),
        (x = 0.0, y = -1.0 * g[7]),
        (x = -1.0 * g[5], y = -1.0 * g[6]),
        (x = -1.0 * g[4], y = 0.0),
        (x = -1.0 * g[2], y = 1.0 * g[3]),
    ]
end

function render_creature(length::Int, stems; dir::Int = 0, oldpos = (0.0, 0.0))
    newdir = mod(dir, 8) + 1
    stem = stems[newdir]
    newpos = (
        oldpos[1] + length * stem.x,
        oldpos[2] + length * stem.y,
    )
    segments = [(start = oldpos, finish = newpos)]
    if length > 1
        append!(segments, render_creature(length - 1, stems; dir = dir + 1, oldpos = newpos))
        append!(segments, render_creature(length - 1, stems; dir = dir - 1, oldpos = newpos))
    end
    return segments
end

function render_radial_creature(length::Int, stems; arms::Int, oldpos = (0.0, 0.0))
    arms < 2 && return render_creature(length, stems; dir = 0, oldpos = oldpos)
    segments = render_creature(length, stems; dir = 0, oldpos = oldpos)
    for k in 1:(arms - 1)
        d = mod(round(Int, 8 * k / arms), 8)
        append!(segments, render_creature(length, stems; dir = d, oldpos = oldpos))
    end
    return segments
end

function calculate_bounds(segments)
    minx, miny = 0.0, 0.0
    maxx, maxy = 0.0, 0.0
    for seg in segments
        x1, y1 = seg.start
        x2, y2 = seg.finish
        minx = min(minx, x1, x2)
        miny = min(miny, y1, y2)
        maxx = max(maxx, x1, x2)
        maxy = max(maxy, y1, y2)
    end
    return (minx = minx, miny = miny, maxx = maxx, maxy = maxy)
end

function draw_line!(img, x1, y1, x2, y2, color; thick::Bool = false)
    h, w = size(img)
    d = max(abs(x2 - x1), abs(y2 - y1))
    steps = max(1, floor(Int, d) + 1)
    for t in range(0.0, 1.0; length = steps + 1)
        x = round(Int, x1 + t * (x2 - x1))
        y = round(Int, y1 + t * (y2 - y1))
        if 1 <= x <= w && 1 <= y <= h
            img[y, x] = color
            if thick
                for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    nx, ny = x + dx, y + dy
                    if 1 <= nx <= w && 1 <= ny <= h
                        img[ny, nx] = color
                    end
                end
            end
        end
    end
end

function render_biomorph_image(genes; active_idx, canvas_size = 150, bg = (1.0, 1.0, 1.0), radial_arms = nothing)
    stems = calculate_stems(genes, active_idx)
    segments =
        radial_arms === nothing || radial_arms < 2 ?
        render_creature(genes[19], stems; dir = 0, oldpos = (0.0, 0.0)) :
        render_radial_creature(genes[19], stems; arms = radial_arms, oldpos = (0.0, 0.0))
    bounds = calculate_bounds(segments)
    width = max(abs(bounds.minx), abs(bounds.maxx)) * 2
    height = max(abs(bounds.miny), abs(bounds.maxy)) * 2
    scale = 120.0 / max(width, height, 1.0)
    img = fill(RGB{N0f8}(bg[1], bg[2], bg[3]), canvas_size, canvas_size)
    color = RGB{N0f8}(genes[16] / 255, genes[17] / 255, genes[18] / 255)
    cx = canvas_size / 2
    cy = canvas_size / 2
    thick = radial_arms !== nothing && radial_arms >= 2
    for seg in segments
        x1, y1 = seg.start
        x2, y2 = seg.finish
        px1 = cx + x1 * scale
        py1 = cy - y1 * scale
        px2 = cx + x2 * scale
        py2 = cy - y2 * scale
        draw_line!(img, px1, py1, px2, py2, color; thick = thick)
    end
    return img
end

function make_tree_target(; canvas_size::Int = 150, bg = (1.0, 1.0, 1.0))
    active_idx = [1, 2, 3, 4, 5, 6, 7]
    genes = [
        6,
        3,
        4,
        4,
        2,
        4,
        6,
        0, 0, 0, 0, 0, 0, 0, 0,
        120,
        220,
        255,
        7,
    ]
    img = render_biomorph_image(genes; active_idx = active_idx, canvas_size = canvas_size, bg = bg)
    return (target = img, genes = genes, active_idx = active_idx)
end

function make_bat_target(; canvas_size::Int = 150, bg = (1.0, 1.0, 1.0))
    active_idx = [1, 2, 3, 4, 5, 6, 7]
    genes = [
        2,
        7,
        7,
        8,
        5,
        6,
        3,
        0, 0, 0, 0, 0, 0, 0, 0,
        235,
        165,
        85,
        8,
    ]
    img = render_biomorph_image(genes; active_idx = active_idx, canvas_size = canvas_size, bg = bg)
    return (target = img, genes = genes, active_idx = active_idx)
end

function estimate_bg_from_corners(img)
    h, w = size(img)
    cr = (img[1, 1], img[1, w], img[h, 1], img[h, w])
    r = sum(Float64(c.r) for c in cr) / 4
    g = sum(Float64(c.g) for c in cr) / 4
    b = sum(Float64(c.b) for c in cr) / 4
    return (r, g, b)
end

function infer_target_color_genes(img; bg_target, fit_size::Int = 150)
    small = RGB.(imresize(img, (fit_size, fit_size)))
    m = prepare_mask(small; fit_size = fit_size, bg = bg_target)
    s_r, s_g, s_b = 0.0, 0.0, 0.0
    cnt = 0
    for i in eachindex(small)
        if m[i] > 0.5
            c = small[i]
            s_r += Float64(c.r)
            s_g += Float64(c.g)
            s_b += Float64(c.b)
            cnt += 1
        end
    end
    if cnt == 0
        for i in eachindex(small)
            c = small[i]
            s_r += Float64(c.r)
            s_g += Float64(c.g)
            s_b += Float64(c.b)
            cnt += 1
        end
    end
    rf = clamp(round(Int, 255 * s_r / cnt), 0, 255)
    gf = clamp(round(Int, 255 * s_g / cnt), 0, 255)
    bf = clamp(round(Int, 255 * s_b / cnt), 0, 255)
    return (rf, gf, bf)
end

function make_target_from_image(path::AbstractString; canvas_size::Int = 150, bg_target = nothing)
    img = RGB.(FileIO.load(path))
    img = imresize(img, (canvas_size, canvas_size))
    bt = bg_target === nothing ? estimate_bg_from_corners(img) : bg_target
    r16, r17, r18 = infer_target_color_genes(img; bg_target = bt, fit_size = canvas_size)
    genes = [fill(0, 15); r16; r17; r18; 7]
    active_idx = collect(1:7)
    return (target = img, genes = genes, active_idx = active_idx, bg_target = bt)
end

function prepare_mask(img; fit_size::Int = 150, eps::Float64 = 1e-6, bg = (1.0, 1.0, 1.0))
    small = RGB.(imresize(img, (fit_size, fit_size)))
    mask = zeros(Float64, size(small))
    br, bgc, bb = bg
    for i in eachindex(small)
        c = small[i]
        dist =
            abs(Float64(c.r) - br) + abs(Float64(c.g) - bgc) + abs(Float64(c.b) - bb)
        mask[i] = dist > eps ? 1.0 : 0.0
    end
    return mask
end

function color_diff(img, target; fit_size::Int = 150, bg = (1.0, 1.0, 1.0))
    a = RGB.(imresize(img, (fit_size, fit_size)))
    b = RGB.(imresize(target, (fit_size, fit_size)))
    ma = prepare_mask(img; fit_size = fit_size, bg = bg) .> 0.5
    mb = prepare_mask(target; fit_size = fit_size, bg = bg) .> 0.5
    mask = ma .| mb
    err = 0.0
    cnt = 0
    for i in eachindex(mask)
        if mask[i]
            err += abs(Float64(a[i].r) - Float64(b[i].r))
            err += abs(Float64(a[i].g) - Float64(b[i].g))
            err += abs(Float64(a[i].b) - Float64(b[i].b))
            cnt += 3
        end
    end
    return cnt == 0 ? 1.0 : err / cnt
end

function color_gene_diff(genes, target_genes)
    return (
        abs(genes[16] - target_genes[16]) +
        abs(genes[17] - target_genes[17]) +
        abs(genes[18] - target_genes[18])
    ) / (3 * 255)
end

function ncc(
    img,
    target;
    fit_size::Int = 150,
    bg = (1.0, 1.0, 1.0),
    bg_target = nothing,
    eps::Float64 = 0.03,
    eps_target = nothing,
)
    bgt = bg_target === nothing ? bg : bg_target
    et = eps_target === nothing ? eps : eps_target
    a = prepare_mask(img; fit_size = fit_size, bg = bg, eps = eps)
    b = prepare_mask(target; fit_size = fit_size, bg = bgt, eps = et)
    av = vec(a)
    bv = vec(b)
    am = mean(av)
    bm = mean(bv)
    num = sum((av .- am) .* (bv .- bm))
    den = sqrt(sum((av .- am) .^ 2) * sum((bv .- bm) .^ 2))
    shape_score = den == 0 ? 1.0 : 1.0 - num / den
    area = sum(a)
    if area < 30
        shape_score += 5.0
    end
    return shape_score
end

function mask_outline(m::AbstractMatrix{Float64})
    h, w = size(m)
    e = zeros(Float64, h, w)
    for y in 1:h, x in 1:w
        m[y, x] < 0.5 && continue
        edge = false
        for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ny, nx = y + dy, x + dx
            if ny < 1 || ny > h || nx < 1 || nx > w
                edge = true
            elseif m[ny, nx] < 0.5
                edge = true
            end
            edge && break
        end
        if edge
            e[y, x] = 1.0
        end
    end
    return e
end

function correlation_shape_score(av::AbstractVector{Float64}, bv::AbstractVector{Float64})
    am = mean(av)
    bm = mean(bv)
    num = sum((av .- am) .* (bv .- bm))
    den = sqrt(sum((av .- am) .^ 2) * sum((bv .- bm) .^ 2))
    return den == 0 ? 1.0 : 1.0 - num / den
end

function outline_shape_score(
    img,
    target;
    fit_size::Int,
    bg,
    bg_target,
    ncc_eps::Float64,
    ncc_eps_target,
)
    bgt = bg_target === nothing ? bg : bg_target
    et = ncc_eps_target === nothing ? ncc_eps : ncc_eps_target
    a = prepare_mask(img; fit_size = fit_size, bg = bg, eps = ncc_eps)
    b = prepare_mask(target; fit_size = fit_size, bg = bgt, eps = et)
    oa = mask_outline(a)
    ob = mask_outline(b)
    return correlation_shape_score(vec(oa), vec(ob))
end

function fitness(
    genes,
    img,
    target,
    target_genes;
    fit_size::Int = 150,
    bg = (1.0, 1.0, 1.0),
    bg_target = nothing,
    ncc_eps::Float64 = 0.03,
    ncc_eps_target = nothing,
    area_match_weight::Float64 = 0.0,
    outline_weight::Float64 = 0.0,
)
    shape_score = ncc(
        img,
        target;
        fit_size = fit_size,
        bg = bg,
        bg_target = bg_target,
        eps = ncc_eps,
        eps_target = ncc_eps_target,
    )
    if area_match_weight > 0 && bg_target !== nothing
        et = ncc_eps_target === nothing ? ncc_eps : ncc_eps_target
        sa = sum(prepare_mask(img; fit_size = fit_size, bg = bg, eps = ncc_eps))
        sb = sum(prepare_mask(target; fit_size = fit_size, bg = bg_target, eps = et))
        shape_score += area_match_weight * abs(sa - sb) / max(sb, 40.0)
    end
    if outline_weight > 0 && bg_target !== nothing
        shape_score += outline_weight * outline_shape_score(
            img,
            target;
            fit_size = fit_size,
            bg = bg,
            bg_target = bg_target,
            ncc_eps = ncc_eps,
            ncc_eps_target = ncc_eps_target,
        )
    end
    color_score = color_gene_diff(genes, target_genes)
    return shape_score + 2.0 * color_score
end

function evolve_biomorph(target, target_genes;
    N = 200,
    fit_size = 150,
    patience = 10,
    max_generations = 300,
    seed = nothing,
    active_idx = nothing,
    bg = (1.0, 1.0, 1.0),
    bg_target = nothing,
    radial_arms = nothing,
    ncc_eps::Float64 = 0.03,
    ncc_eps_target = nothing,
    area_match_weight::Float64 = 0.0,
    outline_weight::Float64 = 0.0,
)
    seed !== nothing && Random.seed!(seed)
    if active_idx === nothing
        active_idx = choose_active_genes(seed)
    end
    initial_pool = [random_genotype() for _ in 1:N]
    initial_imgs = [
        render_biomorph_image(g; active_idx = active_idx, bg = bg, radial_arms = radial_arms) for g in initial_pool
    ]
    initial_scores = [
        fitness(
            g,
            img,
            target,
            target_genes;
            fit_size = fit_size,
            bg = bg,
            bg_target = bg_target,
            ncc_eps = ncc_eps,
            ncc_eps_target = ncc_eps_target,
            area_match_weight = area_match_weight,
            outline_weight = outline_weight,
        ) for (g, img) in zip(initial_pool, initial_imgs)
    ]
    idx0 = argmin(initial_scores)
    parent = copy(initial_pool[idx0])
    parent_img = initial_imgs[idx0]
    parent_score = initial_scores[idx0]
    best_genes = copy(parent)
    best_img = copy(parent_img)
    best_score = parent_score
    history = Float64[parent_score]
    no_improve = 0
    generation = 0
    while generation < max_generations && no_improve < patience
        generation += 1
        candidates = Vector{Vector{Int}}()
        push!(candidates, copy(parent))
        for _ in 1:(N - 1)
            push!(candidates, mutate_genotype(parent, active_idx))
        end
        candidate_imgs = [
            render_biomorph_image(g; active_idx = active_idx, bg = bg, radial_arms = radial_arms) for g in candidates
        ]
        candidate_scores = [
            fitness(
                g,
                img,
                target,
                target_genes;
                fit_size = fit_size,
                bg = bg,
                bg_target = bg_target,
                ncc_eps = ncc_eps,
                ncc_eps_target = ncc_eps_target,
                area_match_weight = area_match_weight,
                outline_weight = outline_weight,
            ) for (g, img) in zip(candidates, candidate_imgs)
        ]
        idx = argmin(candidate_scores)
        parent = copy(candidates[idx])
        parent_img = candidate_imgs[idx]
        parent_score = candidate_scores[idx]
        push!(history, parent_score)
        if parent_score < best_score
            best_score = parent_score
            best_genes = copy(parent)
            best_img = copy(parent_img)
            no_improve = 0
        else
            no_improve += 1
        end
    end
    return (
        best_genes = best_genes,
        best_img = best_img,
        best_score = best_score,
        history = history,
        generations = generation,
        active_idx = active_idx,
    )
end

const BG = (0.0, 0.0, 0.0)

const IMAGE_RADIAL_ARMS = nothing
const IMAGE_NCC_EPS_TARGET = 0.1
const IMAGE_AREA_MATCH_WEIGHT = 0.4
const IMAGE_OUTLINE_WEIGHT = 0.35

tree = make_tree_target(; bg = BG)
bat = make_bat_target(; bg = BG)

result_tree = evolve_biomorph(
    tree.target,
    tree.genes;
    N = 150,
    fit_size = 150,
    patience = 10,
    max_generations = 100,
    seed = 14,
    active_idx = tree.active_idx,
    bg = BG,
)

println()
println("Дерево")
println("Число поколений: ", result_tree.generations)
println("Метрика в начале (поколение 0): ", result_tree.history[1])
println("Метрика у лучшего решения: ", result_tree.best_score)
println("Метрика в последнем поколении: ", result_tree.history[end])

result_bat = evolve_biomorph(
    bat.target,
    bat.genes;
    N = 150,
    fit_size = 150,
    patience = 10,
    max_generations = 100,
    seed = 21,
    active_idx = bat.active_idx,
    bg = BG,
)

println()
println("Летучая мышь")
println("Число поколений: ", result_bat.generations)
println("Метрика в начале (поколение 0): ", result_bat.history[1])
println("Метрика у лучшего решения: ", result_bat.best_score)
println("Метрика в последнем поколении: ", result_bat.history[end])

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
FileIO.save(joinpath(outdir, "target_tree.png"), tree.target)
FileIO.save(joinpath(outdir, "best_tree.png"), result_tree.best_img)
FileIO.save(joinpath(outdir, "target_bat.png"), bat.target)
FileIO.save(joinpath(outdir, "best_bat.png"), result_bat.best_img)

image_path = joinpath(@__DIR__, "images.png")
if isfile(image_path)
    from_img = make_target_from_image(image_path; canvas_size = 150, bg_target = nothing)
    result_img = evolve_biomorph(
        from_img.target,
        from_img.genes;
        N = 150,
        fit_size = 150,
        patience = 10,
        max_generations = 100,
        seed = 42,
        active_idx = from_img.active_idx,
        bg = BG,
        bg_target = from_img.bg_target,
        radial_arms = IMAGE_RADIAL_ARMS,
        ncc_eps = 0.03,
        ncc_eps_target = IMAGE_NCC_EPS_TARGET,
        area_match_weight = IMAGE_AREA_MATCH_WEIGHT,
        outline_weight = IMAGE_OUTLINE_WEIGHT,
    )
    println()
    println("Эталон image.png")
    println("Число поколений: ", result_img.generations)
    println("Метрика в начале (поколение 0): ", result_img.history[1])
    println("Метрика у лучшего решения: ", result_img.best_score)
    println("Метрика в последнем поколении: ", result_img.history[end])
    FileIO.save(joinpath(outdir, "target_image.png"), from_img.target)
    FileIO.save(joinpath(outdir, "best_image.png"), result_img.best_img)
else
    println("Файл image.png не найден, третий прогон пропущен.")
end