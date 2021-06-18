using Printf
@printf "Initializing...\n"

@everywhere include("code/acoustic_solver_parallel.jl")
@everywhere include("code/adjoint_method_ot.jl")
@everywhere include("code/optimization.jl")

@everywhere begin
    @printf "Loading...\n"
    @load "temp_data/data.jld2"
    u0 = copy(c)

    @printf "Preparing optimization function handle...\n"
    eps = 1e-5
    lambda_mix = 1e-10
    M = cost_matrix_1d(t, t; p=2)
    iteration_number = 10
    k_normalize = 5e3
    reg = 1e-4
    reg_m = 1e-1

    eval_fn_ot(x) = obj_func_ot(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position, reg, reg_m, iteration_number, k_normalize; pml_len=20, pml_coef=200)
    eval_grad_ot(x) = grad_ot(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position, reg, reg_m, iteration_number, k_normalize; pml_len=20, pml_coef=200)

    @printf "Initializing optimization ...\n"
    min_value = 0
    max_value = 10
    alpha = 4e7
    iterNum = 20
    rrho = 0.5
    cc = 1e-10
    maxSearchTime = 3
    x0 = reshape(c, Nx*Ny, 1)
end

println("Start nonlinear CG.")
xk, fn = nonlinear_cg(eval_fn_ot, eval_grad_ot, x0, alpha, iterNum, min_value, max_value; rho=rrho, c=cc, maxSearchTime=maxSearchTime, threshold=1e-10);

@save "ex4_ot_result.jld2" xk
