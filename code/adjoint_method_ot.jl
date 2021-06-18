# OT code

function cost_matrix_1d(x, y; p=2)
    Nx = length(x)
    Ny = length(y)

    M = zeros(Nx,Ny)
    for i = 1:Nx
        for j = 1:Ny
            M[i,j] = (x[i] - y[j])^p
        end
    end
    return M
end

function unbalanced_sinkhorn_1d(a, b, M, reg, reg_m; iterMax=100, verbose=true)
    Na = length(a)
    Nb = length(b)
    size_M = size(M)
    if Na != size_M[1] || Nb != size_M[2]
    #     println("size error")
        error("Please check the size of cost matrix.")
    end

    # threshold for the non-zero part
    ind_a = findall(x->x>0, a)
    aa = a[ind_a]
    Naa = length(aa)
    MM = M[ind_a,:]
    K = exp.(-MM./reg)
    fi = reg_m / (reg + reg_m)
    
    u = ones(Naa) ./ Naa
    v = ones(Nb) ./ Nb
    u0 = zeros(Naa)
    v0 = zeros(Nb)
    
    iter = 0
    err = 1
    
    while iter < iterMax
        v = (b ./ (K' * u)) .^ fi
        u = (aa ./ (K * v)) .^ fi
        iter += 1
    end

    # coupling
    T = similar(M)
    @. T[ind_a, :] = (u .* K .* v')

    # gradient w.r.t. a.
    ff = reg*log.(u)
    temp1 = exp.(-ff./reg_m) .- 1
    grad = zeros(Na)
    grad[ind_a] = - reg_m * temp1

    # p-Wasserstein distance ^ p
    a1 = T * ones(Na)
    b1 = T'* ones(Nb)

    loga = a1.*log.(a1./a)
    loga[isnan.(loga)] .= 0
    kla = sum(loga - a1 + a)

    logb = b1.*log.(b1./b)
    logb[isnan.(logb)] .= 0
    klb = sum(logb - b1 + b)

    lTK = T[ind_a,:] .* log.(T[ind_a,:]./K)
    lTK[isnan.(lTK)] .= 0
    lTK = lTK - T[ind_a,:] + K

    dist = reg * sum(lTK) + reg_m * (kla+klb)
    return grad, dist
end


function adj_source_ot_exp(data1, data2, M; reg=1e-3, reg_m=1e2, iterMax=100, k=1);
    adj = similar(data1)
    dist = 0
    @inbounds @views for i = 1:size(data1,2)
        f = exp.(k * data1[:,i])
        g = exp.(k * data2[:,i])
        gg, dd = unbalanced_sinkhorn_1d(f, g, M, reg, reg_m; iterMax=iterMax, verbose=false)
        adj[:,i] = gg
        dist += dd
    end
    return dist, adj
end

function obj_func_ot(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position, reg, reg_m, iterMax, k; pml_len=20, pml_coef=200)
    t = range(0,step=dt,length=Nt)
    M = cost_matrix_1d(t, t);
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        dist, adj_source = adj_source_ot_exp(data_forward, received_data[:,:,ind], M; reg=reg, reg_m=reg_m, iterMax=iterMax, k=k);
        adj_source = nothing
        
        # evaluate the objctive function
        obj_value[ind] = dist
    end

    obj_value = sum(obj_value)
    
    return obj_value
end

function grad_ot(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position, reg, reg_m, iterMax, k; pml_len=20, pml_coef=200)
    t = range(0,step=dt,length=Nt)
    M = cost_matrix_1d(t, t);
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        dist, adj_source = adj_source_ot_exp(data_forward, received_data[:,:,ind], M; reg=reg, reg_m=reg_m, iterMax=iterMax, k=k);
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 1 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3)
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