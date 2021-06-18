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
    eps = 1e-5
    lambda_mix = 1e-10
    M = cost_matrix_1d(t, t; p=2)
    iteration_number = 1000
    k_normalize = 5e3

    compute_adj_source_mixed_tbyt(data_forward, received_data) = adj_source_mixed_tbyt(data_forward, received_data, eps, lambda_mix, M, k_normalize; iter_num=iteration_number)
    eval_fn_ot(x) = eval_obj_fn_mixed_exp(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position, compute_adj_source_mixed_tbyt; pml_len=30, pml_coef=50)
    eval_grad_ot(x) = compute_gradient_mixed_exp_cutoff(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position, compute_adj_source_mixed_tbyt; pml_len=30, pml_coef=50)

    @printf "Initializing optimization ...\n"
    min_value = 0
    max_value = 10
    # alpha = 5e7
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
