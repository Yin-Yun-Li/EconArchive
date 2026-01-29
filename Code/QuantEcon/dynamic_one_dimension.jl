using LaTeXStrings, LinearAlgebra, Plots

### ================== Quant Econ Sample code ==================== ###

# Iterates a function from an initial condition 
function iterate_map(f, x0, T)
    x = zeros(T + 1)
    x[1] = x0
    for t in 2:(T + 1)
        x[t] = f(x[t - 1])
    end
    return x
end

function plot45(f, xmin, xmax, x0, T; num_points = 100, label = L"g(k)",
                xlabel = "k")
    # Plot the function and the 45 degree line
    x_grid = range(xmin, xmax, num_points)
    plt = plot(x_grid, f.(x_grid); xlim = (xmin, xmax), ylim = (xmin, xmax),
               linecolor = :black, lw = 2, label)
    plot!(x_grid, x_grid; linecolor = :blue, lw = 2, label = nothing)

    # Iterate map and add ticks
    x = iterate_map(f, x0, T)
    xticks!(x, [L"%$(xlabel)_{%$i}" for i in 0:T])
    yticks!(x, [L"%$(xlabel)_{%$i}" for i in 0:T])

    # Plot arrows and dashes
    for i in 1:T
        plot!([x[i], x[i]], [x[i], x[i + 1]], arrow = :closed, linecolor = :black,
              alpha = 0.5, label = nothing)
        plot!([x[i], x[i + 1]], [x[i + 1], x[i + 1]], arrow = :closed,
              linecolor = :black, alpha = 0.5, label = nothing)
        plot!([x[i + 1], x[i + 1]], [0, x[i + 1]], linestyle = :dash,
              linecolor = :black, alpha = 0.5, label = nothing)
    end
    plot!([x[1], x[1]], [0, x[1]], linestyle = :dash, linecolor = :black,
          alpha = 0.5, label = nothing)
end

function ts_plot(f, x0, T; xlabel = L"t", label = L"k_t")
    x = iterate_map(f, x0, T)
    plot(0:T, x; xlabel, label)
    plot!(0:T, x; seriestype = :scatter, mc = :blue, alpha = 0.7, label = nothing)
end

### Correction for nonlinear map example

h(k) = 4 * k * (1 - k)
plot45(k -> h(k), 0, 1, 0.1, 6)
ts_plot(h, 0.1, 6)

### Exercise

Base.@kwdef mutable struct Params
    a::Float64 = 1.0
    b::Float64 = 1.0
    xmin::Float64 = 0.0
    xmax::Float64 = 5.0
end


p1 = Params(a=0.5, xmin=-1, xmax=3)
p2 = Params(a=-0.5, xmin=-1, xmax=3)

g(x; p) = p.a*x + p.b

x0 = -0.5
plot45(x -> g(x; p=p1), p1.xmin, p1.xmax, x0, 5)
ts_plot(x -> g(x; p=p1), x0, 10)

plot45(x -> g(x; p=p2), p2.xmin, p2.xmax, x0, 5)
ts_plot(x -> g(x; p=p2), x0, 10)




