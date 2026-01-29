using LinearAlgebra, Statistics, StatsBase, Random, Distributions, StatsFuns
using LaTeXStrings, Plots
using DataFrames


# Exercise 1

a = 0.9
b = 0.1
c = 0.5
mu_0, v_0 = -3.0, 0.6
K = 10
N = 1000000

mu = mu_0
v = v_0

Random.seed!(1234)

mk_table = DataFrame(k=1:K, theoretical=zeros(K), empirical=zeros(K))

for k in 1:K
    mu = a*mu + b
    v = a^2*v + c^2
    xi = rand(Normal(mu, sqrt(v)), N)    
    kth_empirical_moment = moment(xi, k)
    mk_table[k, :empirical] = kth_empirical_moment

    if iseven(k)
        kth_theroetical_moment = sqrt(v)^k * prod(k-1:-2:1)
        mk_table[k, :theoretical] = kth_theroetical_moment
    end
end

mk_table

plot(mk_table.k, [mk_table.theoretical mk_table.empirical], 
    label = ["Theoretical" "Empirical"], 
    xlabel = "Moment order k", 
    ylabel = "Moment value", 
    title = "Theoretical vs Empirical Moments of AR(1) Process", 
    legend = :topleft,
    color = [:teal :crimson],
    lw = 2
)



# Exercise 2

x_grid = range(-0.2, 1.2, length=100)

random_draw = function(; α, β, n=500)
    xi = rand(Beta(α, β), n)
    h = 1.06*std(xi)*n^(-1/5) ### rule of thumb bandwidth
    return (;xi, h)
end

f_kde = function(x; xi, h, n=500)
    return (1/(h*n))*sum(normpdf.( (x .- xi) ./ h))
end

f_kde_2 = function(x; xi, h, n=500)
    kernel_values = pdf.(Normal(0, 1), (x .- xi) ./ h)
    return (1 / (h * n)) * sum(kernel_values)
end

simulate_phi = random_draw(α=2, β=2)

f_kde(0.5; xi = simulate_phi.xi, h = simulate_phi.h)
f_kde_2(0.5; xi = simulate_phi.xi, h = simulate_phi.h)

kde_plot = function(; α, β)
    simulate_phi = random_draw(α=α, β=β)
    f(x) = f_kde(x; xi = simulate_phi.xi, h = simulate_phi.h)
    
    pt = plot(x_grid, f.(x_grid),
       label = "KDE Estimate",
       xlabel = "x",
       ylabel = "Density",
       color = :crimson,
       lw = 2)
     
    plot!(pt, x_grid, pdf.(Beta(α, β), x_grid),
      label = "True Density Beta($α, $β)", 
      legend = :topright,
      color = :teal,
      lw = 2)
    
    return pt
end


parameter_pairs = [(2, 2), (2, 5), (0.5, 0.5)]

plots_array = map(parameter_pairs) do (alpha, beta)
    kde_plot(α=alpha, β=beta)
end

pt1 = plot(plots_array..., 
    layout = (3, 1), 
    size = (600, 800), 
    plot_title = "Kernel Density Estimate vs True Density",
    plot_title_vspan = 0.1
)

savefig(pt1, "kde_beta_plots.png")


### Exercise 3
a = 0.9
b = 0.0
c = 0.1
mu_t, s_t = -3, 0.2

𝛹_t = Normal(mu_t, s_t)
𝛹_prime_t = Normal(a*mu_t + b, sqrt(a^2*s_t^2 + c^2))

x=range(-4,-2,length=100)
pt2 = plot(x, pdf.([𝛹_t 𝛹_prime_t], x), 
    color  = [:slateblue :coral],     
    lw     = 2,                                     
    xlabel = "x",
    ylabel = "Density",
    label = [L"\psi_t" L"\psi_{t+1}"],
)

xt = rand(𝛹_t, 2000)
wt_prime = randn(2000)
xt_prime = @. a*xt + b + c*wt_prime
h=1.06*std(xt_prime)*2000^(-1/5) ### rule of thumb bandwidth

f_kde_2 = function(x; xi, h)
    kernel_values = pdf.(Normal(0, 1), (x .- xi) ./ h)
    return (1 / h) * mean(kernel_values)
end


f(x) = f_kde_2(x; xi = xt_prime, h = h)


plot!(pt2, x, f.(x),
    label = "KDE Estimate " * L"\psi_{t+1}",
    color = :darkgreen,
    lw = 2,
    title = "Transition of Density in AR(1) Process",
    legend = :topleft
)

savefig(pt2, "AR1_density_transition.png")