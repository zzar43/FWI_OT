# this is for the 41-120 iteration with cg algorithm

using Printf
@printf "Initializing...\n"

@everywhere include("code/acoustic_solver_parallel.jl")
@everywhere include("code/adjoint_method.jl")
@everywhere include("code/optimization.jl")

@everywhere begin
    @printf "Loading...\n"
    
    @load "temp_data/data.jld2"
    u0 = copy(c)

    @printf "Preparing optimization function handle...\n"
    eval_fn(x) = obj_func_l2(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    eval_grad(x) = grad_l2(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
    @printf "Initializing optimization ...\n"
    min_value = 0
    max_value = 10
#     alpha = 1e-8
#     alpha = 1e7
    alpha = 9e6
    iterNum = 20
    rrho = 0.2
    cc = 1e-10
    maxSearchTime = 5
    x0 = reshape(c, Nx*Ny, 1);
end

println("Start nonlinear CG.")
xk, fn = nonlinear_cg(eval_fn, eval_grad, x0, alpha, iterNum, min_value, max_value; rho=rrho, c=cc, maxSearchTime=maxSearchTime, threshold=1e-10);

@save "ex4_l2_result.jld2" xk

