# Adjoint state methods functions
# created by: 2020/11/30
include("Mixed_Wasserstein.jl")

# Adjoint source function, with mixed Wasserstein distance, and expoential normalization
# Typical parameter choice for current examples:
# eps = 1e-4
# lambda_mix = 1e-8
# M = cost_matrix_1d(t, t; p=2)
# iteration_number = 1000
# k_normalize = 1e3

function adj_source_mixed_tbyt(data_forward, received_data, eps, lambda_mix, M, k_normalize; iter_num=500)
    adj_source = 0 .* received_data
    dist = 0
    @views @inbounds for i = 1:size(received_data,2)
        b = received_data[:,i]
        a = data_forward[:,i]
        grad_exp, dist_exp = Mixed_Wasserstein_exp(a, b, eps, lambda_mix, M, k_normalize; iter_num=iter_num, verbose=false);
        adj_source[:,i] = grad_exp'
        dist += dist_exp
    end
    return dist, adj_source
end

# compute the gradient of the mixed Wasserstein distance and exp normalization
# define the function handle
# compute_adj_source_mixed_tbyt(data_forward, received_data) = adj_source_mixed_tbyt(data_forward, received_data, eps, lambda_mix, M, k_normalize; iter_num=500)

function compute_gradient_mixed_exp(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position, compute_adj_source_mixed_tbyt; pml_len=30, pml_coef=50)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        dist, adj_source = compute_adj_source_mixed_tbyt(data_forward, received_data[:,:,ind])
        # cutoff
        # adj_source[end-20:end, :] .= adj_source[end-21, :]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3) * dt
        grad[:,:,ind] = grad0
        
        # evaluate the objctive function
        obj_value[ind] = dist
    end
    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]

    obj_value = sum(obj_value)
    grad = reshape(grad, Nx*Ny,1)
    
    return obj_value, grad
end

function compute_gradient_mixed_exp_cutoff(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position, compute_adj_source_mixed_tbyt; pml_len=30, pml_coef=50)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    print("Computing the adjoint source: ")
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        dist, adj_source = compute_adj_source_mixed_tbyt(data_forward, received_data[:,:,ind])
        # cutoff
        # adj_source[end-20:end, :] .= adj_source[end-21, :]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3) * dt
        grad[:,:,ind] = grad0
        
        # evaluate the objctive function
        obj_value[ind] = dist
        print(ind)
        print(". ")
    end
    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]
    grad[1:16,:] .= 0

    obj_value = sum(obj_value)
    grad = reshape(grad, Nx*Ny,1)
    println(" Done.")
    
    return obj_value, grad
end

# evaluate the objective function
function eval_obj_fn_mixed_exp(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position, compute_adj_source_mixed_tbyt; pml_len=30, pml_coef=50)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        data_forward = acoustic_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        dist, adj_source = compute_adj_source_mixed_tbyt(data_forward, received_data[:,:,ind])
        # evaluate the objctive function
        obj_value[ind] = dist
    end

    obj_value = sum(obj_value)
    return obj_value
end

# L2 adjoint method

function obj_func_l2(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=200)
    c = reshape(c, Nx, Ny)
    
    data = multi_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    
    return 0.5*norm(received_data-data,2)^2
end

function grad_l2(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=200)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        @views adj_source = data_forward - received_data[:,:,ind]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 1 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3)
        grad[:,:,ind] = grad0
        
        # evaluate the objctive function
        obj_value[ind] = 0.5 * norm(data_forward-received_data[:,:,ind],2).^2
    end
    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]
    grad[1:16,:] .= 0

    obj_value = sum(obj_value)
    grad = reshape(grad, Nx*Ny,1)
    return obj_value, grad
end
