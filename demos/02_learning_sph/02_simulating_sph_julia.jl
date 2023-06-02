using NPZ, Plots, Statistics, LinearAlgebra
using LaTeXStrings

using InteractiveUtils, BenchmarkTools
using StaticArrays



const Decimal = Float64


# np.random.seed(42)
const EXPERIMENET_ID = "3D"
const DATAFOLDER = "./demos/02_learning_sph/data/" * EXPERIMENET_ID *"/"
# make_dir(DATAFOLDER)



const IC = "taylor-green"         ## Taylor-Green IC
# const IC = "random"             ## Random IC

const T = 100
const T_SAVE = 1   #initial time for saving
# PRINT_EVERY = ceil(Int, T/10)
const PRINT_EVERY = 1
const DURATION = 15   ## Seconds

const MAX_VEL = Decimal(1.0)
const c = Decimal(0.9157061661168617)
const h = Decimal(0.2)  ##TODO Rebmember
const α = Decimal(0.45216843078299573)
const β = Decimal(0.3346233846532608)
const γ = Decimal(1.0)                     ## TODO Equal 7 in the paper (see fig 1)
const θ = Decimal(0.00430899795067121)
const dt = Decimal(0.4 * h / c)

const L = Decimal(2*pi)    ## domain limit accors all axis

const D = 3
const LOG_RES = 4  ##TODO Rebmember



##

# """
# All required functions
# """


# const σ = Decimal(1. / (pi * (h^3)))

# function W(r, h, σ)
#     q = r / h   
#     if (q > 2)   return Decimal(0.)   end
#     if (q > 1)   return σ * (2 - q)^3 / 4   end
#     return σ * (1 - Decimal(1.5) * q * q * (1 - q / 2))
# end

# function H(r, h, σ)
#     q = r / h
#     if (q > 2)   return Decimal(0.)   end
#     if (q > 1)   return -3 * σ * (2 - q)^2 / (4 * h * r)   end
#     return σ * (-3 + 9 * q / 4) / (h^2)
# end


## TODO this kernel is super fast because it takes the True route within the IF
const σ = Decimal(315 / (208 * pi * (h^6)))

function W(r, h, σ)
    q = r / h
    if (q <= 2) return σ * ((2/3) - (9*q^2/8) + (19*q^3/24) - (5*q^4/32))
    else return Decimal(0.0) end
end

function H(r, h, σ)
    q = r / h
    if (q <= 2) return σ * (- (9/4) + (19*q/8) - (5*q^2/8)) / (h^2)
    else return Decimal(0.0) end
end



function meshgrid(xlim, ylim, zlim, logres)
    particles = Iterators.product((xlim/(2^logres)):(xlim/(2^logres)):xlim, 
                            (ylim/(2^logres)):(ylim/(2^logres)):ylim, 
                            (zlim/(2^logres)):(zlim/(2^logres)):zlim)
    return stack(vec(collect.(particles)), dims=1)
end


function P(ρ, c, γ)
  return c^2 * (ρ^γ - 1) / γ
end

function compute_c(ρ, c, γ)
  return c * ρ^((γ - 1)/2)
end

function compute_Π(Xij, Vij, ρi, ρj, h, c, γ, α, β)
    c_bar = (compute_c(ρi, c, γ) + compute_c(ρj, c, γ)) / 2
    ρ_bar = (ρi + ρj) / 2

    if Xij'*Vij < 0
        μ = h * (Xij'*Vij) / (Xij'*Xij + Decimal(0.01)*(h^2))
        Πij = (-α*c_bar*μ + β*(μ^2)) / ρ_bar
    else
        Πij = Decimal(0.)
    end

    return Πij

end

function compute_densities!(ρ, dists, h, m, σ)
    N = size(ρ)[1]
    fill!(ρ, Decimal(0.))

    for i in 1:N
        ρ[i] = m * W(Decimal(0.), h, σ)
        for dist in dists[i]
            ρ[i] += m * W(dist, h, σ)
        end
    end
end


function compute_acc_forces!(F, X, V, ρ, neighbors, dists, D, L, h, m, σ, θ, c, γ, α, β)

    N = size(ρ)[1]
    fill!(F, Decimal(0.))

    for i in 1:N
        vali = P(ρ[i], c, γ) / (ρ[i]^2)
        nb_neighbors = size(neighbors[i])[1]

        for j_ in 1:nb_neighbors
            j = neighbors[i][j_]

            ## Compute artificial viscosity
            Xij = X[i, :] - X[j, :]
            Vij = V[i, :] - V[j, :]
            for d in 1:D
                Xij_abs = abs(Xij[d])      ## TODO Optimize this
                if Xij_abs > L/2
                    Xij[d] = sign(Xij[d]) * (Xij_abs - L)
                end
                # while (Xij[d] > L/2.) Xij[d] -= L end
                # while (Xij[d] < -L/2.) Xij[d] += L end
            end

            Πij = compute_Π(Xij, Vij, ρ[i], ρ[j], h, c, γ, α, β)

            valj = P(ρ[j], c, γ) / (ρ[j]^2)
        
            ## Add up all forces
            F[i, :] += -m*(vali + valj + Πij) * H(dists[i][j_], h, σ) .* Xij ## TODO External forcing?

        end

    end

    # ke = Decimal(0.5) * mean(ρ .* sum(V.^2, dims=2))
    ke = mean(ρ .* sum(V.^2, dims=2)) / 2
    F += θ .* V / ke
    # F += θ .* (V .- mean(V, dims=1)) / (2 * ke)    ## TODO Change to the above as in the paper

    println()

end


function compute_cell_id(i, j, k, n_cells)
    return i + n_cells * (j-1 + n_cells*(k-1))
end


function make_cells_for_nn_search(L, h, D)
    n = ceil(Int, L / (2*h))

    cells = Matrix{Int}(undef, (n^D, 3^D))
    neighbors = Vector{Int}(undef, 3^D)

    for k in 1:n
        for j in 1:n
            for i in 1:n

                counter = 1
                for k_ in k-1:k+1
                    for j_ in j-1:j+1
                        for i_ in i-1:i+1
                                neighbors[counter] = compute_cell_id(i_%n+1, j_%n+1, k_%n+1, n)

                                counter += 1
                        end
                    end
                end

                cell_id = compute_cell_id(i,j,k, n)
                cells[cell_id, :] = neighbors

            end
        end
    end

    return cells

end

function find_cell(x, h, n)
    i = floor(Int, x[1]/(2*h)) + 1
    j = floor(Int, x[2]/(2*h)) + 1
    k = floor(Int, x[3]/(2*h)) + 1

    return compute_cell_id(i, j, k, n)

end

function distance(x, y, L)
    # diff = abs.(x - y)
    # diff_min = min.(diff, L .- diff)
    # return sqrt(sum(diff_min.^2))

    diff1 = abs(x[1] - y[1])
    diff2 = abs(x[2] - y[2])
    diff3 = abs(x[3] - y[3])

    return sqrt(min(diff1, L-diff1)^2 +
                min(diff2, L-diff2)^2 +
                min(diff3, L-diff3)^2)

    # return sqrt(diff1^2 + diff2^2 + diff3^2)
end

function fixed_radius_nn_search!(neighbors, distances,points_to_cell, cells_to_points, cells, X, L, h)

    N = size(X)[1]
    n = ceil(Int, L / (2*h))
    nb_cells = n^D      ## size(cells)[1]

    for i in 1:N
        points_to_cell[i] = find_cell(X[i,:], h, n)
    end

    for cell_id in 1:nb_cells
        cells_to_points[cell_id] = findall(k->k==cell_id, points_to_cell)
    end


    for i in 1:N
        neighbor_cells = cells[points_to_cell[i], :]

        neighs_i = Int[]
        dists_i = Decimal[]

        for cell_id in neighbor_cells
            for j in cells_to_points[cell_id]
                dist = distance(X[i,:], X[j,:], L)
                if (dist < 2*h) && (j != i) 
                    push!(neighs_i, j)
                    push!(dists_i, dist)
                end
            end
        end

        neighbors[i] = neighs_i[:]
        distances[i] = dists_i[:]
    end

end



function random_ic(points, vmag)
    noise = Decimal(0.0005) .* (rand(Decimal, size(points)) .- Decimal(0.5))

    X = points + noise

    V = vmag * randn(Decimal, size(points))
    return X, V
end


function taylor_green_ic(points, vmag)
    x = points[:,1]
    y = points[:,2]
    z = points[:,3]

    u = vmag .* sin.(x) .* cos.(y) .* cos.(z)
    v = -vmag .* cos.(x) .* sin.(y) .* cos.(z)
    w = zero(x)

    return points, hcat(u, v, w)
end



## 




function simulate_flow()


    particles = meshgrid(L, L, L, LOG_RES)
    N = size(particles)[1]
    m = L^D / N       ## constant mass of each particle


    println("Weakly Compressible SPH")
    println(" -Number of particles: ", N)
    println(" -Number of time steps: ", T)
    println(" -Initial condition: ", IC)


    trajs = Array{Decimal}(undef, (T+1, N, D))
    vels = Array{Decimal}(undef, (T+1, N, D))
    rhos = Array{Decimal}(undef, (T+1, N))

    X = Array{Decimal}(undef, (N, D))
    V = Array{Decimal}(undef, (N, D))
    F = zeros(Decimal, (N, D))
    ρ = zeros(Decimal, (N))

    ## Initial conditions
    if IC == "random"
        X, V = random_ic(particles, MAX_VEL)
    elseif IC == "taylor-green"
        X, V = taylor_green_ic(particles, MAX_VEL)
    else
        error("Unknown initial condition")
    end
    X = mod.(X, L)


    trajs[1,:,:] = X
    vels[1,:,:] = V
    rhos[1,:] = ρ

    cells = make_cells_for_nn_search(L, h, D)

    neighbors = Vector{Vector{Int}}(undef, N)
    distances = Vector{Vector{Decimal}}(undef, N)
    points_to_cell = Vector{Int}(undef, N)
    cells_to_points = Vector{Vector{Int}}(undef, size(cells)[1])

    energy = Float64[]

    for t in 1:T
        if t % PRINT_EVERY == 0
            println("simulating time step: ", t)

            println("max F = ", maximum(F[:]))
            println("min F = ", minimum(F[:]))
        
            println("max V = ", maximum(V[:]))
            println("min V = ", minimum(V[:]))
        
            println("max rho = ", maximum(ρ[:]))
            println("min rho = ", minimum(ρ[:]))
        end

        ## Initial density
        fixed_radius_nn_search!(neighbors, distances,points_to_cell, cells_to_points, cells, X, L, h)

        compute_densities!(ρ, distances, h, m, σ)
        compute_acc_forces!(F, X, V, ρ, neighbors, distances, D, L, h, m, σ, θ, c, γ, α, β)

        V += dt .* F / 2
        X += dt * V
        X = mod.(X, L)

        fixed_radius_nn_search!(neighbors, distances,points_to_cell, cells_to_points, cells, X, L, h)

        compute_acc_forces!(F, X, V, ρ, neighbors, distances, D, L, h, m, σ, θ, c, γ, α, β)

        V += dt .* F / 2

        ## Compute kinetic energy
        ke = mean(ρ .* sum(V.^2, dims=2)) / 2
        push!(energy, ke)

        trajs[t+1, :, :] = X
        vels[t+1, :, :] = V
        rhos[t+1, :] = ρ

    end

    display(plot(energy, label="KE"))

    return trajs, vels, rhos

end



trajs, vels, rhos = @time simulate_flow();


##



function visualise_trajectory(pos, sim_time=5)
    gr(size=(1000,800))

    # sim_path = DATAFOLDER*"trajs_N$(N)_T$(T)_h$(h)_$(IC)_c$(c)_α$(α)_β$(β)_θ$(θ).mp4"
    sim_path = DATAFOLDER*"trajectory.mp4"
    println("**************** Visualising the particle flow ***************")
    N = size(pos)[2]

    #theme(:juno)
    n_2 = round(Int, N/2); m_s = 1.75
    anim = @animate for i ∈ 1:(T+1)
        Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
        title = "Simulated WCSPH: N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms = m_s)

        Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red", ms = m_s)
    end
    gif(anim, sim_path, fps = ceil(Int, T/sim_time))
    println("****************  Visualisation COMPLETE  *************")
end



visualise_trajectory(trajs, DURATION)




##


function save_data_files(trajs, vels, rhos, t_save)
    println(" ****************** Saving data files ***********************")
    pos_path = DATAFOLDER*"trajs_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ).npy"
    vel_path = DATAFOLDER*"/vels_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ).npy"
    rho_path = DATAFOLDER*"/rhos_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ).npy"

    npzwrite(pos_path, trajs[t_save:end,:,:])
    npzwrite(vel_path, vels[t_save:end,:,:])
    npzwrite(rho_path, rhos[t_save:end,:])
end

# save_data_files(trajs, vels, rhos, T_SAVE)



##

