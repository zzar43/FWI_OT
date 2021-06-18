# This is the basic sikhorn algorithm
# created by: 2020/11/30

using LinearAlgebra
# ==========================================
# Basic test functions
# ==========================================
function gauss_func(t, b, c)
    y = exp.(-(t.-b).^2 ./ (2*c^2));
    return y
end

function gaussian_2d(X,Y,center,sigma)
    g = exp.(-(X.-center[1]).^2 ./ (sigma[1]^2) -(Y.-center[2]).^2 ./ (sigma[2]^2))
    g = g ./ maximum(g)
    return g
end

function sin_func(t, omega, phi)
    return sin.(2*pi*omega*(t .- phi));
end

function ricker_func(t, t0, sigma)
    t = t.-t0;
    f = (1 .- t.^2 ./ sigma.^2) .* exp.(- t.^2 ./ (2 .* sigma.^2));
    return f
end

function ricker_2d(X,Y,center,sigma)
    g = (1 .- (X.-center[1]).^2 ./ (sigma[1]^2) .- (Y.-center[2]).^2 ./ (sigma[2]^2)) .* exp.(-(X.-center[1]).^2 ./ (sigma[1]^2) -(Y.-center[2]).^2 ./ (sigma[2]^2))
    g = g ./ maximum(g)
    return g
end

# ==========================================
# 1d case
# ==========================================
# make 1d cost matrix
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

# TP: with transport plan
function sinkhorn_1d_TP_with_0(a, b, eps, M; threshold=1e-5, iter_num=0, verbose=false)
    grad = 0 .* a
    iteration = true
    dist0 = 0
    dist1 = 0
    
    Na = length(a)
    Nb = length(b)

    ind_a = findall(x->x>0, a)
    aa = a[ind_a]
    Naa = length(aa)
    MM = M[ind_a,:]
    K = exp.(-MM./eps)

    u = ones(Naa) ./ Naa
    v = ones(Nb) ./ Nb
    T = 0 .* M
    aa_temp = 0 .* a
    b_temp = 0 .* b
    
    if iter_num != 0
        @inbounds @views for iter = 1:iter_num
            aa_temp = K * v
            @. u = aa / aa_temp
            b_temp = K' * u
            @. v = b / b_temp
            if any(isnan.(u))
                error("Nan happens. Increase eps.")
            end
        end
        @. T[ind_a, :] = (u * K * v')
        dist1 = T .* M
        dist1 = sum(dist1)
    else
        iter = 0
        @inbounds @views while iteration == true
            aa_temp = K * v
            @. u = aa / aa_temp
            b_temp = K' * u
            @. v = b / b_temp
            if any(isnan.(u))
                error("Nan happens. Increase eps.")
            end
            @. T[ind_a, :] = (u * K * v') * MM
            dist1 = sum(T)
            if abs(dist1/dist0-1) < threshold
                if verbose == true
                    print("Sinkhorn converges after ")
                    print(iter)
                    println(" iterations.")
                end
                iteration = false
            end
            dist0 = copy(dist1)
            iter += 1
        end
        @. T[ind_a, :] = (u * K * v')
    end
    
    # gradient w.r.t. a. Normalized with sum(grad) = 0
    grad[ind_a] = eps*log.(u) .- (eps/Naa) * sum(log.(u))

    return T, grad, dist1
end

function sinkhorn_1d_TP(a, b, eps, M; threshold=1e-5, iter_num=0, verbose=false)
    grad = 0 .* a
    iteration = true
    dist0 = 0
    dist1 = 0
    
    Na = length(a)
    Nb = length(b)
    u = ones(Na) ./ Na
    v = ones(Nb) ./ Nb
    K = exp.(-M ./ eps)
    T = 0 .* M
    aa = 0 .* a
    bb = 0 .* b
    
    if iter_num != 0
        @inbounds @views for iter = 1:iter_num
            aa = K * v
            @. u = a / aa
            bb = K' * u
            @. v = b / bb
            if any(isnan.(u))
                error("Nan happens. Increase eps.")
            end
        end
        @. T = (u * K * v')
        dist1 = T .* M
        dist1 = sum(dist1)
    else
        iter = 0
        @inbounds @views while iteration == true
            aa = K * v
            if any(isnan.(a))
                stop
            end
            @. u = a / aa
            bb = K' * u
            @. v = b / bb
            if any(isnan.(u))
                error("Nan happens. Increase eps.")
            end
            @. T = (u * K * v') * M
            dist1 = sum(T)
            if abs(dist1/dist0-1) < threshold
                if verbose == true
                    print("Sinkhorn converges after ")
                    print(iter)
                    println(" iterations.")
                end
                iteration = false
            end
            dist0 = copy(dist1)
            iter += 1
        end
        @. T = (u * K * v')
    end
    
    # gradient w.r.t. a. Normalized with sum(grad) = 0
    grad = eps*log.(u) .- (eps/Na) * sum(log.(u))
    
    return T, grad, dist1
end


# ==========================================
# 2d case
# ==========================================

# make 2d cost matrix
# This function can be used in the case that rerrange the 2d images into 1d vectors
function cost_matrix_2d(x, y; p=2)
    # x, y: row vectors, for the domain
    Nx = length(x)
    Ny = length(y)
    X = repeat(x, 1, Ny)
    Y = repeat(y', Nx, 1)
    X = reshape(X, Nx*Ny, 1)
    Y = reshape(Y, Nx*Ny, 1)
    M = zeros(Nx*Ny, Nx*Ny)
    for i = 1:Nx*Ny
        for j = 1:Nx*Ny
            M[i,j] = sqrt((X[i]-X[j])^2 + (Y[i]-Y[j])^2)^p
        end
    end
    return M
end
# make 2d cost matrix: convolution form
function cost_matrix_2d_conv(x, y; p=2)
    C1 = zeros(Nx, Nx)
    for i = 1:Nx
        for j = 1:Nx
            C1[i,j] = (x[i] - x[j])^p
        end
    end

    C2 = zeros(Ny, Ny)
    for i = 1:Ny
        for j = 1:Ny
            C2[i,j] = (y[i] - y[j])^p
        end
    end

    return C1, C2
end

# convolution form of sinkhorn algorithm for 2d images
# M1, M2 are the C1, C2 from function cost_matrix_2d_conv
function sinkhorn_2d(a, b, eps, M1, M2; iter_num=100)
    Nx, Ny = size(a)
    u = ones(Nx, Ny)
    v = ones(Nx, Ny)
    K1 = exp.(-M1/eps)
    K2 = exp.(-M2/eps)

#     @fastmath for iter = 1:iter_num
    for iter = 1:iter_num
        aa = K1 * v * K2'
        @. u = a / aa
        bb = K1' * u * K2
        @. v = b / bb
        if any(isnan.(u))
            print("Iteration: ")
            println(iter)
            error("Nan happens. Increase eps.")
        end
    end
    
    grad = eps*log.(u) .- (eps/(Nx*Ny)) * sum(log.(u))
    
    dist = 0
    i = 0
    j = 0
     @fastmath @inbounds @simd for j1 = 1:Ny
        for i1 = 1:Ny
            for j2 = 1:Nx
                for i2 = 1:Nx
                    i = i2 + (i1-1)*Nx
                    j = j2 + (j1-1)*Nx
                    dist += u[i] * K2[i1,j1] * K1[i2,j2] * v[j] * (M2[i1,j1] + M1[i2,j2])
                end
            end
        end
    end
    
    return grad, dist
end