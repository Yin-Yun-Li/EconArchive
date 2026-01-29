# Note: This file mainly come from QuantEcon.jl. Minor modifications have been made for the purpose of exercises.
#=
Computes quantities related to the Gaussian linear state space model

    x_{t+1} = A x_t + C w_{t+1}

        y_t = G x_t + H v_t

The shocks {w_t} and {v_t} are iid and N(0, I)

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date : 2014-07-28

References
----------

TODO: Come back and update to match `LinearStateSpace` type from py side
TODO: Add docstrings

https://lectures.quantecon.org/jl/kalman.html

=#


### =================================================== ###

using QuantEcon, Distributions, LinearAlgebra, Statistics, LaTeXStrings, Plots, StatsPlots
const ScalarOrArray = Union{Number, AbstractArray}

@doc raw"""
    LSS

A type that describes the Gaussian Linear State Space Model
of the form:

```math
    x_{t+1} = A x_t + C w_{t+1} \\

    y_t = G x_t + H v_t
```

where ``{w_t}`` and ``{v_t}`` are independent and standard normal with dimensions
`k` and `l` respectively.  The initial conditions are ``\mu_0`` and ``\Sigma_0`` for ``x_0
\sim N(\mu_0, \Sigma_0)``. When ``\Sigma_0=0``, the draw of ``x_0`` is exactly ``\mu_0``.

# Fields

- `A::Matrix`: Part of the state transition equation.  It should be `n x n`.
- `C::Matrix`: Part of the state transition equation.  It should be `n x m`.
- `G::Matrix`: Part of the observation equation.  It should be `k x n`.
- `H::Matrix`: Part of the observation equation.  It should be `k x l`.
- `k::Int`: Dimension.
- `n::Int`: Dimension.
- `m::Int`: Dimension.
- `l::Int`: Dimension.
- `mu_0::Vector`: This is the mean of initial draw and is of length `n`.
- `Sigma_0::Matrix`: This is the variance of the initial draw and is `n x n` and
  also should be positive definite and symmetric.

"""
mutable struct LSS{TSampler<:MVNSampler}
    A::Matrix
    C::Matrix
    G::Matrix
    H::Matrix
    k::Int
    n::Int
    m::Int
    l::Int
    mu_0::Vector
    Sigma_0::Matrix
    dist::TSampler
end



function LSS(A::ScalarOrArray, C::ScalarOrArray, G::ScalarOrArray, H::ScalarOrArray,
             mu_0::ScalarOrArray,
             Sigma_0::Matrix=zeros(size(G, 2), size(G, 2)))
    k = size(G, 1)
    n = size(G, 2)
    m = size(C, 2)
    l = size(H, 2)

    # coerce shapes
    A = reshape(vcat(A), n, n)
    C = reshape(vcat(C), n, m)
    G = reshape(vcat(G), k, n)
    H = reshape(vcat(H), k, l)

    mu_0 = reshape([mu_0;], n)

    dist = MVNSampler(mu_0,Sigma_0)
    LSS(A, C, G, H, k, n, m, l, mu_0, Sigma_0, dist)
end

# make kwarg version; user input
function LSS(A::ScalarOrArray, C::ScalarOrArray, G::ScalarOrArray;
             H::ScalarOrArray=zeros(size(G, 1)),
             mu_0::Vector=zeros(size(G, 2)),
             Sigma_0::Matrix=zeros(size(G, 2), size(G, 2)))
    return LSS(A, C, G, H, mu_0, Sigma_0) ### previous more complicated function
end

### DGP
function simulate(lss::LSS, ts_length=100)
    x = Matrix{Float64}(undef, lss.n, ts_length)
    x[:, 1] = rand(lss.dist)
    w = randn(lss.m, ts_length - 1)
    v = randn(lss.l, ts_length)
    for t=1:ts_length-1
        x[:, t+1] = lss.A * x[:, t] .+ lss.C * w[:, t]
    end
    y = lss.G * x + lss.H * v

    return x, y
end

### Exercise 1 
SODE = LSS([1 0 0; 1.1 0.8 -0.8; 0 1 0], # A
           [0.0 0.0 0.0]', # C
           [0.0 1.0 0.0];  # G
           mu_0 = ones(3))  
x,y = simulate(SODE, 50)
plot(vec(y), 
     color = :cornflowerblue, 
     linewidth = 2, 
     title = "Second Order Difference Equation: Simulated Path of Observations",
     titlefontsize = 10,
     xlabel = "Time", ylabel = L"y_t", legend = false)


### Exercise 2
𝝓_1, 𝝓_2, 𝝓_3, 𝝓_4 = 0.5, -0.2, 0.0, 0.5 
σ = 0.2
UVA = LSS([𝝓_1 𝝓_2 𝝓_3 𝝓_4;
            1 0 0 0;
            0 1 0 0;
            0 0 1 0], # A
           [σ 0.0 0.0 0.0]', # C
           [1.0 0.0 0.0 0.0];  # G
           mu_0 = ones(4))  
x,y = simulate(UVA, 200)
plot(vec(y), 
     color = :cornflowerblue, 
     linewidth = 2, 
     title = "Univariate Autoregressive Processes: Simulated Path of Observations",
     titlefontsize = 10,
     xlabel = "Time", ylabel = L"y_t", legend = false)



### =================================================== ###

@doc raw"""
    replicate(lss, t, num_reps)

Simulate `num_reps` observations of ``x_T`` and ``y_T`` given ``x_0 \sim N(\mu_0, \Sigma_0)``.

# Arguments

- `lss::LSS`: An instance of the Gaussian linear state space model.
- `t::Int = 10`: The period that we want to replicate values for.
- `num_reps::Int = 100`: The number of replications we want.

# Returns

- `x::Matrix`: An `n x num_reps` matrix, where the j-th column is the j_th
  observation of ``x_T``.
- `y::Matrix`: A `k x num_reps` matrix, where the j-th column is the j_th
  observation of ``y_T``.

"""
function replicate(lss::LSS, t::Integer, num_reps::Integer=100)
    x = Matrix{Float64}(undef, lss.n, num_reps)
    v = randn(lss.l, num_reps)
    for j=1:num_reps
        x_t, _ = simulate(lss, t+1)
        x[:, j] = x_t[:, end]
    end

    y = lss.G * x + lss.H * v
    return x, y
end

replicate(lss::LSS; t::Integer=10, num_reps::Integer=100) =
    replicate(lss, t, num_reps)

### Exercise 3

T, I = 50, 20
rep_matrix = zeros(T, I)
tran_vec = zeros(T)
σ = 0.1
A = [𝝓_1 𝝓_2 𝝓_3 𝝓_4 
       1 0 0 0
       0 1 0 0
       0 0 1 0]
G = [1.0 0.0 0.0 0.0]
μ_0 = ones(4)
uva = LSS(A, [σ 0.0 0.0 0.0]', G; mu_0 = μ_0)  

### Ensemble Average
for i in 1:I
    res = simulate(uva, T)
    rep_matrix[:,i] =  res[2]
end

### Longrun Average
μt = μ_0
for t in 1:T
    μt_prime = A*μt
    tran_vec[t] = only(G*μt_prime)
    μt = μt_prime
end

plot(rep_matrix, 
     color = :cornflowerblue, 
     alpha = 0.3,             
     linewidth = 1.5,       
     label = "",         
     title = "",
     xlabel = "Time",
     ylabel = L"y_t",
     legend = :topright
)

ensemble_avg = mean(rep_matrix, dims=2)

plot!(ensemble_avg, 
      color = :darkblue, 
      linewidth = 2, 
      label = "Ensemble Average " * L"(\overline{y_t})",
)


plot!(tran_vec, 
      color = :darkgreen, 
      linewidth = 2, 
      label = "Longrun Average " * L"(G \mu_t)"
)



### =================================================== ###

struct LSSMoments
    lss::LSS
end


function Base.iterate(L::LSSMoments, state=(copy(L.lss.mu_0),
                                            copy(L.lss.Sigma_0)))
    A, C, G, H = L.lss.A, L.lss.C, L.lss.G, L.lss.H
    mu_x, Sigma_x = state

    mu_y, Sigma_y = G * mu_x, G * Sigma_x * G' + H * H'

    # Update moments of x
    mu_x2 = A * mu_x
    Sigma_x2 = A * Sigma_x * A' + C * C'

    return ((mu_x, mu_y, Sigma_x, Sigma_y), (mu_x2, Sigma_x2))
end


@doc raw"""
    moment_sequence(lss)

Create an iterator to calculate the population mean and
variance-covariance matrix for both ``x_t`` and ``y_t``, starting at
the initial condition `(self.mu_0, self.Sigma_0)`.  Each iteration
produces a 4-tuple of items `(mu_x, mu_y, Sigma_x, Sigma_y)` for
the next period.

# Arguments

- `lss::LSS`: An instance of the Gaussian linear state space model.

# Returns

- `iterator`: An iterator that yields 4-tuples `(mu_x, mu_y, Sigma_x, Sigma_y)` for each period.

"""
moment_sequence(lss::LSS) = LSSMoments(lss)

@doc raw"""
    stationary_distributions(lss; max_iter, tol)

Compute the moments of the stationary distributions of ``x_t`` and
``y_t`` if possible.  Computation is by iteration, starting from the
initial conditions `lss.mu_0` and `lss.Sigma_0`.

# Arguments

- `lss::LSS`: An instance of the Gaussian linear state space model.
- `;max_iter::Int = 200`: The maximum number of iterations allowed.
- `;tol::Float64 = 1e-5`: The tolerance level one wishes to achieve.

# Returns

- `mu_x::Vector`: Represents the stationary mean of ``x_t``.
- `mu_y::Vector`: Represents the stationary mean of ``y_t``.
- `Sigma_x::Matrix`: Represents the var-cov matrix.
- `Sigma_y::Matrix`: Represents the var-cov matrix.

"""
function stationary_distributions(lss::LSS; max_iter=200, tol=1e-5)
    !is_stable(lss.A) ? error("Cannot compute stationary distribution because the system is not stable.") : nothing

    # Initialize iteration
    m = moment_sequence(lss)
    mu_x, mu_y, Sigma_x, Sigma_y = first(m)

    i = 0
    err = tol + 1.0

    for (mu_x1, mu_y, Sigma_x1, Sigma_y) in m
        i > max_iter && error("Convergence failed after $i iterations")
        i += 1
        err_mu = maximum(abs, mu_x1 - mu_x)
        err_Sigma = maximum(abs, Sigma_x1 - Sigma_x)
        err = max(err_Sigma, err_mu)
        mu_x, Sigma_x = mu_x1, Sigma_x1

        if err < tol && i > 1
            # return here because of how scoping works in loops.
            return mu_x1, mu_y, Sigma_x1, Sigma_y
        end
    end
end


### Exercise 4
μx, μy, Σx, Σy = stationary_distributions(uva)
T, I = 100, 80
rep_matrix = zeros(T, I)
σ = 0.1
A = [𝝓_1 𝝓_2 𝝓_3 𝝓_4 
       1 0 0 0
       0 1 0 0
       0 0 1 0]
G = [1.0 0.0 0.0 0.0]
stationary_uva = LSS(A, [σ 0.0 0.0 0.0]', G; mu_0 = μx)  

for i in 1:I
    res = simulate(stationary_uva, T)
    rep_matrix[:,i] =  res[2]
end


time_points = [10, 50, 75]

p1 = plot(rep_matrix, 
    label = "", 
    linealpha = 0.4, 
    linewidth = 1.2,
    color_palette = :viridis, 
    grid = true,
    xlabel = "Time",
    ylabel = L"y_t",
    title = "Simulation Dynamic Paths & Cross-sectional Patterns",
    size = (800, 500)
)

vline!(p1, time_points, 
    color = :black, 
    linewidth = 1.5, 
    label = ""
)

for t in time_points

    x_vals = fill(t, I) 
    y_vals = rep_matrix[t, :]
    
    scatter!(p1, x_vals, y_vals,
        color = :black,      
        alpha = 0.6,        
        markersize = 4,      
        markerstrokewidth = 0, 
        label = ""
    )
end

xticks!(p1, time_points, [L"T", L"T^\prime", L"T^{\prime\prime}"])
display(p1)




p2 = plot(layout = (1, 3), size = (900, 300), legend = nothing)
p2 = plot(title = "Cross-sectional Distributions Over Time",
         xlabel = "Value", 
         ylabel = "Density",
         legend = :topright) 

for t in time_points
    data_slice = rep_matrix[t, :]

    density!(p2, data_slice, 
             label = "t = $t", 
             linewidth = 2, 
             fill = (0, 0.1)) 
end

display(p2)




function geometric_sums(lss::LSS, bet, x_t)
    !is_stable(lss) ? error("Cannot compute geometric sum because the system is not stable.") : nothing
    # I = eye(lss.n)
    S_x = (I - bet .* lss.A) \ x_t
    S_y = lss.G * S_x
    return S_x, S_y
end

@doc doc"""
    is_stable(lss)

Test for stability of linear state space system.
First removes the constant row and column.

# Arguments

- `lss::LSS`: The linear state space system.

# Returns

- `stable::Bool`: Whether or not the system is stable.

"""
function is_stable(lss::LSS)

    # Get version of A without constant row/column
    A = remove_constants(lss)

    # Check for stability
    stable = is_stable(A)
    return stable

end

@doc doc"""
    remove_constants(lss)

Finds the row and column, if any,  that correspond to the constant
term in a `LSS` system and removes them to get the matrix that needs
to be checked for stability.

# Arguments

- `lss::LSS`: The linear state space system.

# Returns

- `A::Matrix`: The matrix A with constant row and column removed.

"""
function remove_constants(lss::LSS)
    # Get size of matrix
    A = lss.A
    n, m = size(A)
    @assert n==m

    # Sum the absolute values of each row -> Do this because we
    # want to find rows that the sum of the absolute values is 1
    row_sums_to_one = (vec(sum(abs, A, dims = 2) .- 1.0)) .< 1e-14
    is_ii_one = map(i->abs(A[i, i] - 1.0) < 1e-14, 1:n)
    not_constant_index = .!(row_sums_to_one .& is_ii_one)

    return A[not_constant_index, not_constant_index]
end
