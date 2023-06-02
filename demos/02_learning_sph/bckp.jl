using NPZ, Plots, Statistics, LinearAlgebra
using LaTeXStrings

using InteractiveUtils, BenchmarkTools
using StaticArrays

Decimal = Float64

##

# """
# All required functions
# """


function meshgrid(xlim, ylim, zlim, halfres)
    particles = Iterators.product((xlim/(2^halfres)):(xlim/(2^halfres)):xlim, 
                            (ylim/(2^halfres)):(ylim/(2^halfres)):ylim, 
                            (zlim/(2^halfres)):(zlim/(2^halfres)):zlim)
    return stack(vec(collect.(particles)), dims=1)
end


# println(@code_llvm meshgrid(1.0, 1.0, 1.0, 2))
# methods(meshgrid)




function W(r, h, σ)
    q = r / h   
    if (q > 2)   return 0.   end
    if (q > 1.)   return σ * (2. - q)^3 / 4.   end
    return σ * (1. - 1.5 * q * q * (1. - q / 2.))
end

function H(r, h, σ)
    q = r / h
    if (q > 2.)   return 0.   end
    if (q > 1.)   return -3. * σ * (2. - q)^2 / (4. * h * r)   end
    return σ * (-3. + 9. * q / 4.) / h^2
end

function P(ρ, c, γ)
  return c^2 * (ρ^γ - 1) / γ
end

function compute_c(ρ, c, γ)
  return c * ρ^((γ - 1)/2)
end

function compute_Π(Xij, Vij, ρi, ρj, h, c, γ, α, β)
    c_bar = (compute_c(ρi, c, γ) + compute_c(ρj, c, γ)) / 2.
    ρ_bar = (ρi + ρj) / 2.

    if Xij'*Vij .< 0
      μ = h * (Xij'*Vij) / (Xij'*Xij + 0.01*(h^2))
      return (-α*c_bar*μ + β*(μ^2)) / ρ_bar
    else
      return oftype(h, 0.)
    end
end

function compute_densities!(ρ, dists, h, m, σ)
    N = size(ρ)[1]
    for i in 1:N
        ρ[i] = m * W(oftype(h, 0.), h, σ)
        for dist in dists[i]
            ρ[i] += m * W(dist, h, σ)
        end
    end
    return ρ
end


function compute_acc_forces!(F, X, V, ρ, neighbors, dists, D, L, h, m, σ, θ, c, γ, α, β)

    N = size(ρ)[1]

    for i in 1:N
        vali = P(ρ[i], c, γ) / (ρ[i]^2)
        nb_neighors = size(neighbors[i])[1]

        for j_ in 1:nb_neighors
            j = neighbors[i][j_]

            ## Compute artificial viscosity
            Xij = X[i, :] - X[j, :]
            Vij = V[i, :] - V[j, :]
            for d in 1:D 
                Xij_abs = abs(Xij[d])      ## TODO Optimize this
                if Xij_abs > L/2
                    Xij[d] = sign(Xij[d]) * (Xij_abs - L)
                end
            end

            Πij = compute_Π(Xij, Vij, ρ[i], ρ[j], h, c, γ, α, β)
        
            valj = P(ρ[j], c, γ) / (ρ[j]^2)
        
            ## Add up all forces
            F[i, :] += -m*(vali + valj + Πij) * H(dists[i][j_], h, σ) .* Xij ## TODO External forcing?

        end

    end

    # ke = Float32(0.5) * mean(ρ .* sum(V.^2, dims=2))
    ke = mean(ρ .* sum(V.^2, dims=2)) / 2
    # F += θ * V / ke
    F += θ .* (V .- mean(V, dims=1)) / (2 * ke)    ## TODO Change to the above as in the paper

    return F

end

# function apply_periodic_bcs(X)
#     return mod.(X, L)
# end


function compute_cell_id(i, j, k, n_cells)
    return i + n_cells * (j-1 + n_cells*(k-1))
end


function make_cells_for_nn_search(L, h, D)
    n = ceil(Int, L / h)

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
    i = ceil(Int, x[1]/h)
    j = ceil(Int, x[2]/h)
    k = ceil(Int, x[3]/h)

    return compute_cell_id(i, j, k, n)

end

function distance(x, y, L)
    diff = abs.(x - y)
    diff_min = min.(diff, L .- diff)
    return sqrt(sum(diff_min.^2))
end

function fixed_radius_nn_search(X, L, h, cells)

    N = Int(size(X)[1])
    n = ceil(Int, L / h)
    nb_cells = n^3      ## size(cells)[1]

    ## TODO: Make these a parameter. We cannot assign them each time
    neighbors = Vector{Vector{Int}}(undef, N)
    distancess = Vector{Vector{Float32}}(undef, N)
    points_to_cell = Vector{Int}(undef, N)
    cells_to_points = Vector{Vector{Int}}(undef, N)


    for i in 1:N
        points_to_cell[i] = find_cell(X[i,:], h, n)
    end

    for cell_id in 1:nb_cells
        cells_to_points[cell_id] = findall(i->i==cell_id, points_to_cell)
    end


    for i in 1:N
        neighbor_cells = cells[points_to_cell[i], :]

        neigh_i = Int[]
        dists_i = Float32[]

        for cell_id in neighbor_cells
            for j in cells_to_points[cell_id]
                dist = distance(X[i,:], X[j,:], L)
                if (j != i) && (dist < h)
                    push!(neigh_i, j)
                    push!(dists_i, dist)
                end
            end
        end

        neighbors[i] = neigh_i
        distancess[i] = dists_i
    end

    return neighbors, distancess    ## TODO pass these in to be mutated

end



function random_ic(points, vmag)
    X = points + 0.0005 * (rand(size(points)) - 0.5)
    V = vmag * randn(size(points))
    return X, V
end

function taylor_green_ic(points, vmag)
    x = points[:,1]
    y = points[:,2]
    z = points[:,3]

    u = -vmag * sin.(x) .* cos.(y) .* cos.(z)
    v =  vmag * cos.(x) .* sin.(y) .* cos.(z)
    w = zero(x)

    return points, hcat(u, v, w)

end









## 




function main()

    # np.random.seed(42)
    EXPERIMENET_ID = "3D"
    DATAFOLDER = "./demos/02_learning_sph/data/" * EXPERIMENET_ID *"/"
    # make_dir(DATAFOLDER)

    #params:
    IC = "taylor-green"         ## Taylor-Green IC
    # IC = "random"             ## Random IC

    T = 5
    T_SAVE = 1   #initial time for saving
    PRINT_EVERY = ceil(Int, T/10)
    DURATION = 15   ## Seconds

    MAX_VEL = Float32(1.0)   #initial magnitude of Taylor-Green velocity
    c = Float32(0.9157061661168617)
    h = Float32(0.2)
    α = Float32(0.45216843078299573)
    β = Float32(0.3346233846532608)
    γ = Float32(1.0)                     ## TODO Equal 7 in the paper (see fig 1)
    θ = Float32(0.00430899795067121)
    dt = Float32(0.4 * h / c)

    L = Float32(2*pi)    ## domain limit accors all axis

    D = 3
    HALF_RES = 5;  ## produces grid of 2^halfres x 2^halfres x 2^halfres number of particles


    particles = meshgrid(L, L, L, HALF_RES)
    N = Int(size(particles)[1])
    m = L^D / N       ## constant mass of each particle


    σ = Float32(1. / (pi * (h^3)))


    println("Weakly Compressible SPH")
    println(" -Number of particles: ", N)
    println(" -Number of time steps: ", T)


    trajs = Array{Float32}(undef, (T+1, N, D))
    vels = Array{Float32}(undef, (T+1, N, D))
    rhos = Array{Float32}(undef, (T+1, N))

    X = Array{Float32}(undef, (N, D))
    V = Array{Float32}(undef, (N, D))
    F = zeros(Float32, (N, D))
    ρ = zeros(Float32, (N))

    ## Initial conditions
    if IC == "random"
        X, V = random_ic(particles, MAX_VEL)
    elseif IC == "taylor-green"
        X, V = taylor_green_ic(particles, MAX_VEL)
    else
        error("Unknown initial condition")
    end


    trajs[1,:,:] = X
    vels[1,:,:] = V
    rhos[1,:] = ρ


    ## Initial density

    cells = make_cells_for_nn_search(L, h, D)

    for t in 2:T
        if t % PRINT_EVERY == 0
            println("simulating time step: = ", t)
        end

        ## Initial density
        neighbors, dists = fixed_radius_nn_search(X, L, h, cells)


        ρ = compute_densities!(ρ, dists, h, m, σ)
        F = compute_acc_forces!(F, X, V, ρ, neighbors, dists, D, L, h, m, σ, θ, c, γ, α, β)

        V[:, :] += dt .* F / 2
        X[:, :] += dt * V
        X = mod.(X, L)

        ρ = compute_densities!(ρ, dists, h, m, σ)
        F = compute_acc_forces!(F, X, V, ρ, neighbors, dists, D, L, h, m, σ, θ, c, γ, α, β)

        V[:, :] += dt .* F / 2

        trajs[t, :, :] = X
        vels[t, :, :] = V
        rhos[t, :] = ρ

    end






    function simulate(pos, sim_time=5)
        gr(size=(1000,800))
        sim_path = DATAFOLDER*"trajs_N$(N)_T$(T)_h$(h)_$(IC)_c$(c)_α$(α)_β$(β)_θ$(θ).mp4"
        println("**************** Visualising the particle flow ***************")
        #theme(:juno)
        n_2 = round(Int,N/2); m_s = 1.75
        anim = @animate for i ∈ 1:(T+1)
             println("sim time = ", i)
             Plots.scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], pos[i, 1:n_2, 3],
             title = "Simulated WCSPH: N=$(N)", xlims = [0, 2*pi], ylims = [0,2*pi], zlims = [0,2*pi], legend = false, ms = m_s)
             Plots.scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], pos[i, (n_2+1):end, 3], color = "red", ms = m_s)
        end
        gif(anim, sim_path, fps = ceil(Int, T/sim_time))
        println("****************  Visualisation COMPLETE  *************")
    end

    simulate(trajs, DURATION)

    function save_data_files(trajs, vels, rhos, t_save)
        println(" ****************** Saving data files ***********************")
        pos_path = DATAFOLDER*"trajs_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ).npy"
        vel_path = DATAFOLDER*"/vels_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ).npy"
        rho_path = DATAFOLDER*"/rhos_N$(N)_T$(T)_dt$(dt)_ts$(t_save)_h$(h)_$(IC)_θ$(θ).npy"

        npzwrite(pos_path, trajs[t_save:end,:,:])
        npzwrite(vel_path, vels[t_save:end,:,:])
        npzwrite(rho_path, rhos[t_save:end,:])
    end

    save_data_files(trajs, vels, rhos, T_SAVE)


end



@time main()


##

