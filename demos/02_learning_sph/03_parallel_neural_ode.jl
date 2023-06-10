"""
					Neural ODEs in Parallel
   
"""


##

#### Neural ODEs from Chris and DiffEqFlux

using ComponentArrays, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, Plots

rng = Random.default_rng()
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Lux.Chain(x -> x.^3,
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

theta_init = ComponentArray(p=p,u0=u0)

function predict_neuralode(theta)
  Array(prob_neuralode(theta.u0, theta.p, st)[1])
end

function loss_neuralode(theta)
    pred = predict_neuralode(theta)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (theta, l, pred; doplot = true)
  println(l)
  # plot current prediction against data
  if doplot
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    display(plot(plt))
  end
  return false
end

callback(theta_init, loss_neuralode(theta_init)...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((theta, x) -> loss_neuralode(theta), adtype)
optprob = Optimization.OptimizationProblem(optf, theta_init)

result_neuralode = Optimization.solve(optprob,
                                       ADAM(0.05),
                                       callback = callback,
                                       maxiters = 300)





##




using ComponentArrays, Lux, Random, Plots

rng = Random.default_rng()
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Lux.Chain(x -> x.^3,
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2), first)
p, st = Lux.setup(rng, dudt2)

# dudt2(u0)

using Flux
model = Flux.Chain(x -> x.^3,
			Flux.Dense(2 => 50, tanh),
			Flux.Dense(50 => 2))

# optim = Flux.setup(Flux.Adam(0.01), model)
optim = Flux.setup(Flux.Descent(10.), model)

# println(model(u0))


params = Flux.params(model)
println(params[1])

grads = []

for i in params
	# println(size(i))
	push!(grads, zero(i) .+ 1000.)
end
# for i in grads
# 	println(size(i))
# end

typeof(grads)
gradients = Flux.params(grads)
# println(optim)


gradients = gradient(()->model(u0)[1], params)

# grags = ones(params)
new_opt, new_params = Flux.update!(optim, params, gradients);

# println(params[1])
println("SAME")
println(new_params[1])

# println(model(u0))


# println(typeof(new_params))





# flat_params = Float32[]
# # shapes = []
# for i in params
# 	flat_params = vcat(flat_params, vec(i))
# 	# println(size(vec(i)))
# end
# # size(flat_params)


# flat_grads = zero(flat_params) .+ 10.
# flat_gradients = Flux.params(flat_grads...)


# new_opt, new_params = Flux.update!(optim, flat_params, flat_gradients);

# # println(flat_params)
# # println("\n\nSAME\n\n")
# # println(new_params)

# # println(model(u0))

# # model[2].weight
# model[2].bias .= 10.

# # model


##




# v = vcat([], [3,4])

# ### Loss function
# function L(z_1, z_true)
# 	return sum(abs2, z_1 - z_true)
# end

# ### dL/dz_1
# function dLdz1(z_1, z_true)
# 	return 2*(z_1 - z_true)
# end

# function augmented_dynamics(y, t, θ)
	
# end