# Note: This file mainly come from QuantEcon.jl. Minor modifications have been made for the purpose of exercises.
#=
Implements the Kalman filter for a linear Gaussian state space model.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-07-29

References
----------

https://julia.quantecon.org/tools_and_techniques/kalman.html
=#

using QuantEcon, LinearAlgebra, Distributions, StatsFuns, Plots, StatsPlots, LaTeXStrings, Random

"""
    Kalman

Represents a Kalman filter for a linear Gaussian state space model.

# Fields

- `A`: State transition matrix.
- `G`: Observation matrix.
- `Q`: State noise covariance matrix.
- `R`: Observation noise covariance matrix.
- `k`: Number of observed variables.
- `n`: Number of state variables.
- `cur_x_hat`: Current estimate of state mean.
- `cur_sigma`: Current estimate of state covariance.
"""

mutable struct Kalman
    A
    G
    Q
    R
    k
    n
    cur_x_hat
    cur_sigma
end


# Initializes current mean and cov to zeros
function Kalman(A, G, Q, R)
    k = size(G, 1)
    n = size(G, 2)
    xhat = n == 1 ? zero(eltype(A)) : zeros(n)
    Sigma = n == 1 ? zero(eltype(A)) : zeros(n, n)
    return Kalman(A, G, Q, R, k, n, xhat, Sigma)
end


# Step 1: Start the current period with prior
"""
    set_state!(k, x_hat, Sigma)

Set the current state estimate of the Kalman filter.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.
- `x_hat`: The state mean estimate.
- `Sigma`: The state covariance estimate.

# Returns

- `nothing`: This function modifies the Kalman filter in place.
"""
function set_state!(k::Kalman, x_hat, Sigma)
    k.cur_x_hat = x_hat
    k.cur_sigma = Sigma
    nothing
end


# Step 2: Compute the filtering distribution p_t(x|y) ~ N(̂x^F, Σ^F) via Baysien rule
## x for true states; y for observations
@doc raw"""
    prior_to_filtered!(k, y)

Updates the moments (`cur_x_hat`, `cur_sigma`) of the time ``t`` prior to the
time ``t`` filtering distribution, using current measurement ``y_t``.
The updates are according to

```math
    \hat{x}^F = \hat{x} + \Sigma G' (G \Sigma G' + R)^{-1}
                    (y - G \hat{x}) \\

    \Sigma^F = \Sigma - \Sigma G' (G \Sigma G' + R)^{-1} G
               \Sigma
```

# Arguments

- `k::Kalman`: An instance of the Kalman filter.
- `y`: The current measurement.

# Returns

- `nothing`: This function modifies the Kalman filter in place.

"""
function prior_to_filtered!(k::Kalman, y)
    # simplify notation
    G, R = k.G, k.R
    x_hat, Sigma = k.cur_x_hat, k.cur_sigma

    # and then update
    if k.k > 1
        reshape(y, k.k, 1)
    end
    A = Sigma * G'
    B = G * Sigma * G' + R
    M = A / B
    k.cur_x_hat = x_hat + M * (y .- G * x_hat)
    k.cur_sigma = Sigma - M * G * Sigma
    nothing
end


# Step 3: Compute the predictive distribution p_{t+1}(x), from the filtering distribution p_{t}(x|y)
"""
    filtered_to_forecast!(k)

Updates the moments of the time ``t`` filtering distribution to the
moments of the predictive distribution, which becomes the time
``t+1`` prior.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.

# Returns

- `nothing`: This function modifies the Kalman filter in place.

"""
function filtered_to_forecast!(k::Kalman)
    # simplify notation
    A, Q = k.A, k.Q
    x_hat, Sigma = k.cur_x_hat, k.cur_sigma

    # and then update
    k.cur_x_hat = A * x_hat  ### mean
    k.cur_sigma = A * Sigma * A' + Q  ### var-cov
    nothing
end



"""
    update!(k, y)

Updates `cur_x_hat` and `cur_sigma` given array `y` of length `k`.  The full
update, from one period to the next.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.
- `y`: An array representing the current measurement.

# Returns

- `nothing`: This function modifies the Kalman filter in place.

"""
function update!(k::Kalman, y)
    prior_to_filtered!(k, y)
    filtered_to_forecast!(k)
    nothing
end

# Step 4: Iteration (Use update function, which combine step 2 & 3)

### ================ Exercise 1 ================ ###
θ = 10.0
k1 = Kalman(1,1,0,1)
set_state!(k1, 8.0, 1.0)  
density_table = zeros(200,5)

xgrid = range(θ-5, θ+2, length=200)
for t in (1:5)
    mu, vcv = k1.cur_x_hat, k1.cur_sigma
    density_table[:,t] = normpdf.(mu, sqrt(vcv), xgrid)

    y = θ + randn()
    update!(k1, y)
end

labels = [L"t=0" L"t=1" L"t=2" L"t=3" L"t=4"]
plot(xgrid, density_table, label = labels, lw = 2,
     title = L"First 5 Predictive Distributions given $\theta=10$")


### ================ Exercise 2 ================ ###
ϵ = 0.1
T = 600
k1 = Kalman(1,1,0,1)
set_state!(k1, 8.0, 1.0)  
z_t = zeros(T)
for t in (1:T)
    mu, vcv = k1.cur_x_hat, k1.cur_sigma
    z_t[t] = 1 - (normcdf(mu, sqrt(vcv), θ+ϵ) - normcdf(mu, sqrt(vcv), θ-ϵ))

    y = θ + randn()
    update!(k1, y)
end

plot(1:T, z_t, 
    lw = 2,
    title = "Convergence of Kalman Filter",
    label = L"z_t",             
    color = :cornflowerblue,    
    fill = (0, 0.3, :cornflowerblue),
    grid = :true,              
    gridalpha = 0.2             
)


### ================ Exercise 3 ================ ###
Random.seed!(2568)
x = zeros(2)
A = [0.5 0.4
     0.6 0.3] 
G = Matrix{Float64}(I,2,2)
Q = 0.3*G
R = 0.5*G
Σ0 = [0.9 0.3
      0.3 0.9]

k2 = Kalman(A,G,Q,R)
set_state!(k2, [8.0, 8.0], Σ0) 
T = 50
e1 = zeros(T)
e2 = similar(e1)
xt_prime = x

for t in (1:T)
    xt = xt_prime

    ### Kalman filter
    mu, vcv = k2.cur_x_hat, k2.cur_sigma
    y = G*xt + rand(MvNormal(zeros(k2.k), k2.R)) 
    update!(k2, y)
    
    ### conditional expectation
    Ax = A*xt
    xt_prime = Ax + rand(MvNormal(zeros(k2.n), k2.Q)) 

    e1[t] = norm(xt_prime - k2.cur_x_hat, 2)^2 
    e2[t] = norm(xt_prime - Ax, 2)^2 
end


plot(1:T, [e1 e2], 
     label = ["Kalman filter error" "conditional expectation error"],
     color = [:darkblue :goldenrod],  
     linewidth = 2, 
     alpha = 0.8)



### ================ Exercise 4 ================ ###
𝜙 = [0.1, 0.3, 0.5, 0.7]
A = [0.5 0.4
     0.6 0.3] 
G = Matrix{Float64}(I,2,2)
R = 0.5*G
Σ0 = [0.9 0.3
      0.3 0.9]
T = 50

function run_simulation(phi, A, G, R, Sigma0, T)
    Q = phi * G
    
    kn = Kalman(A, G, Q, R)
    set_state!(kn, [8.0, 8.0], Sigma0)

    x = zeros(2)
    e1 = zeros(T)
    e2 = zeros(T)
    
    for t in 1:T
        ### Kalman filter
        y = G * x + rand(MvNormal(zeros(kn.k), kn.R))
        update!(kn, y)
        ### CEF
        Ax = A * x
        x_next = Ax + rand(MvNormal(zeros(kn.n), kn.Q))
        
        e1[t] = norm(x_next - kn.cur_x_hat, 2)^2
        e2[t] = norm(x_next - Ax, 2)^2
        
        x = x_next
    end
    return (e1, e2) 
end

results = [run_simulation(phi, A, G, R, Σ0, T) for phi in 𝜙]

plot_list = []

for (i, phi) in enumerate(𝜙)
    e1, e2 = results[i]

    p = plot(1:T, [e1 e2],
             label = ["Kalman Error" "Expectation Error"],
             color = [:slateblue :goldenrod],
             linewidth = 2, alpha = 0.8,
             title = latexstring("\\phi = $(phi)"),
             legend = (i == 2 ? :topright : false), 
             grid = :y)
             
    push!(plot_list, p)
end

plot(plot_list..., layout = (2, 2), size = (800, 600))






# Supplementary

"""
    stationary_values(k)

Compute the stationary covariance matrix and Kalman gain for the filter.

# Arguments

- `k::Kalman`: An instance of the Kalman filter.

# Returns

- `Sigma_inf`: The stationary state covariance matrix.
- `K_inf`: The stationary Kalman gain matrix.
"""
function stationary_values(k::Kalman)
    # simplify notation

    A, Q, G, R = k.A, k.Q, k.G, k.R

    # solve Riccati equation, obtain Kalman gain
    Sigma_inf = solve_discrete_riccati(A', G', Q, R)
    K_inf = A * Sigma_inf * G' * inv(G * Sigma_inf * G' .+ R)
    return Sigma_inf, K_inf
end


