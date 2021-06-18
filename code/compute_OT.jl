using LinearAlgebra

# # ===============================
# # Old version
# # make cost matrix M
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

# # 1d sinkhorn function
# function sinkhorn_1d(a, b, eps, M; threshold=1e-5)
#     grad = 0 .* a
#     iteration = true
#     dist0 = 0
#     dist1 = 0

#     Na = length(a)
#     Nb = length(b)
#     u = ones(Na) ./ Na
#     v = ones(Nb) ./ Nb
#     K = exp.(-M ./ eps)
#     T = 0 .* M
#     aa = 0 .* a
#     bb = 0 .* b
    
#     iter = 0
#     @inbounds @views while iteration == true
#         aa = K * v
#         @. u = a / aa
#         bb = K' * u
#         @. v = b / bb
#         if any(isnan.(u))
#             error("Nan happens. Increase eps.")
#         end
#         @. T = (u * K * v') * M
#         dist1 = sum(T)
#         if abs(dist1/dist0-1) < threshold
#             println(iter)
#             iteration = false
#         end
#         dist0 = copy(dist1)
#         iter += 1
#     end

#     # gradient w.r.t. a. Normalized with sum(grad) = 0
#     grad = eps*log.(u) .- (eps/Na) * sum(log.(u))
    
#     return grad, dist1
# end

# function sinkhorn_1d_TP(a, b, eps, M; threshold=1e-5)
#     grad = 0 .* a
#     iteration = true
#     dist0 = 0
#     dist1 = 0

#     Na = length(a)
#     Nb = length(b)
#     u = ones(Na) ./ Na
#     v = ones(Nb) ./ Nb
#     K = exp.(-M ./ eps)
#     T = 0 .* M
#     aa = 0 .* a
#     bb = 0 .* b
    
#     iter = 0
#     @inbounds @views while iteration == true
#         aa = K * v
#         @. u = a / aa
#         bb = K' * u
#         @. v = b / bb
#         if any(isnan.(u))
#             error("Nan happens. Increase eps.")
#         end
#         @. T = (u * K * v') * M
#         dist1 = sum(T)
#         if abs(dist1/dist0-1) < threshold
#             println(iter)
#             iteration = false
#         end
#         dist0 = copy(dist1)
#         iter += 1
#     end
    
#     @. T = (u * K * v')
#     # gradient w.r.t. a. Normalized with sum(grad) = 0
#     grad = eps*log.(u) .- (eps/Na) * sum(log.(u))
    
#     return T, grad, dist1
# end
# # ===============================

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