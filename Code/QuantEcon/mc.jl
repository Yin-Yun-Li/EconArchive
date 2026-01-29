using LinearAlgebra, Statistics, Distributions, Random
using Plots, Graphs, LaTeXStrings, GraphRecipes
using DataFrames

##### ============= Ergodicity ============= #####

### Quantecon template

function mc_sample_path(P; init, sample_size = 10_000)
    @assert size(P)[1] == size(P)[2] # square required
    N = size(P)[1] # should be square

    # create vector of discrete RVs for each row
    dists = [Categorical(P[i, :]) for i in 1:N]

    # setup the simulation
    X = fill(0, sample_size) # allocate memory, or zeros(Int64, sample_size)
    X[1] = init # set the initial state

    for t in 2:sample_size
        dist = dists[X[t - 1]] # get discrete RV from last state's transition distribution
        X[t] = rand(dist) # draw new value
    end
    return X
end


function mc_sample_avg(sequence::Vector)
    xbar = ones(Float64, length(sequence))

    for m in 1:length(sequence)
        mu_m = mean(sequence[1:m] .==1) ### alternative: cumsum(X .== 1) ./ (1:N)
        xbar[m] = mu_m
    end
    return xbar
end

α, β = 0.1, 0.1
P = [1-α α; β 1-β]

p = α /(α+β)

X_1 = mc_sample_path(P, init = 1); 
X_2 = mc_sample_path(P, init = 2)

X_1_bar = mc_sample_avg(X_1)
X_2_bar = mc_sample_avg(X_2)

err1, err2 = X_1_bar .- p, X_2_bar .- p

plot(1:10_000, [err1 err2], 
    label = [L"X_0 = 1" L"X_0 = 2"], 
    xlabel = "m", 
    ylabel = L"\overline{X}_m-p", 
    title = "Ergodicity", 
    legend = :topright,
    color = [:teal :crimson],
    lw = 1.2
)

hline!([0], 
    label = "", 
    color = :purple, 
    linestyle = :dash, 
    lw = 2)


##### ============= Page rank ============= #####


web_graph_data = Dict(
    'a' => ['d','f'],
    'b' => ['j','k','m'],
    'c' => ['c','g','j','m'],
    'd' => ['f','h','k'],
    'e' => ['d','h','l'],
    'f' => ['a','b','j','l'],
    'g' => ['b','j'],
    'h' => ['d','g','l','m'],
    'i' => ['g','h','n'],
    'j' => ['e','i','k'],
    'k' => ['n'],
    'l' => ['m'],
    'm' => ['g'],
    'n' => ['c','j','m']
)


# Sort nodes to ensure consistent matrix indexing (a=1, b=2, etc.)
nodes = sort(collect(keys(web_graph_data)))
index_map = Dict(c => i for (i, c) in enumerate(nodes))
n = length(nodes)

# Digraph
g = SimpleDiGraph(n)

for (source, targets) in web_graph_data
    i = index_map[source]
    for target in targets
        if haskey(index_map, target) # ensure target exist in web_graph_data
            j = index_map[target]
            add_edge!(g, i, j)
        end
    end
end

graphplot(g,
    names = nodes,       
    size = (800, 600),                  
    arrows = true,             
    nodeshape = :circle,       
    nodecolor = :lightgreen,
    edgecolor = :black,         
    method = :spring,  # try :spectral
    nodesize = 0.2,
    curvature = 0.01,       
    fontsize = 10,
    shorten = 0.1              
)



# Build Stochastic Matrix P directly
P = zeros(n, n)

for (source, targets) in web_graph_data
    i = index_map[source]
    k = length(targets) # number of outbound links
    
    # Assign equal probability to each outbound link
    for target in targets
        j = index_map[target]
        P[i, j] = 1.0 / k 
    end
end


### Helper function from Quantecon

function stationary_distributions(P)
    n, m = size(P)
    @assert n == m "Transition matrix must be square"
    ev = eigen(Matrix{Float64}(P'))
    idxs = findall(λ -> isapprox(λ, 1), ev.values) ### Perron-Frobenius thm. ∃ an eigenvalue λ = 1
    @assert !isempty(idxs) "No unit eigenvalue found for the transition matrix"
    dists = Vector{Vector{Float64}}()
    for idx in idxs
        v = real.(ev.vectors[:, idx]) ### Perron-Frobenius thm. Non-negative Real Numbers (probability)
        push!(dists, vec(v ./ sum(v))) 
    end
    return dists
end

r = stationary_distributions(P)[1] # If P is irreducible, then P has only one stationary distribution.

df = DataFrame( node = nodes, rank = r )
@show sort!(df, :rank, rev=true)