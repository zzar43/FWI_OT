# functions
# Ricker function for source
function source_ricker(center_fre, center_time, t)
    x = (1 .- 2*pi^2 .* center_fre^2 .* (t.-center_time).^2) .* exp.(-pi^2*center_fre^2 .* (t .- center_time).^2);
end

# extension of the domain
function extend_vel(vel, Nx_pml, Ny_pml, pml_len)
    vel_ex = zeros(Nx_pml, Ny_pml);
    @views vel_ex[pml_len+1:end-pml_len, pml_len+1:end-pml_len] .= vel;
    for i = 1:pml_len
        vel_ex[i,:] = vel_ex[pml_len+1,:];
        vel_ex[end-i+1,:] = vel_ex[end-pml_len,:];
        vel_ex[:,i] = vel_ex[:,pml_len+1];
        vel_ex[:,end-i+1] = vel_ex[:,end-pml_len];
    end
    return vel_ex
end

# differential operator
function dx_forward!(dA::AbstractMatrix, A::AbstractMatrix, h)
    Nx, Ny = size(A)
    @boundscheck (Nx,Ny) == size(dA) || throw(BoundsError())
    @inbounds for j ∈ 2:Ny-1
        for i ∈ 2:Nx-1
            dA[i,j] = (A[i+1,j]-A[i,j]) / h
        end
    end
end

function dy_forward!(dA::AbstractMatrix, A::AbstractMatrix, h)
    Nx, Ny = size(A)
    @boundscheck (Nx,Ny) == size(dA) || throw(BoundsError())
    @inbounds for j ∈ 2:Ny-1
        for i ∈ 2:Nx-1
            dA[i,j] = (A[i,j+1]-A[i,j]) / h
        end
    end
end

function dx_backward!(dA::AbstractMatrix, A::AbstractMatrix, h)
    Nx, Ny = size(A)
    @boundscheck (Nx,Ny) == size(dA) || throw(BoundsError())
    @inbounds for j ∈ 2:Ny-1
        for i ∈ 2:Nx-1
            dA[i,j] = (A[i,j]-A[i-1,j]) / h
        end
    end
end

function dy_backward!(dA::AbstractMatrix, A::AbstractMatrix, h)
    Nx, Ny = size(A)
    @boundscheck (Nx,Ny) == size(dA) || throw(BoundsError())
    @inbounds for j ∈ 2:Ny-1
        for i ∈ 2:Nx-1
            dA[i,j] = (A[i,j]-A[i,j-1]) / h
        end
    end
end

# main program
function acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    # reshape
    c = reshape(c, Nx, Ny)
    rho = reshape(rho, Nx, Ny)
    
    if size(source_position,2) != 2
        error("Please check the size of source_position, which should be a Ns times 2 matrix.")
    end
    if size(receiver_position,2) != 2
        error("Please check the size of receiver_position, which should be a Nr times 2 matrix.")
    end
    if size(source,1) != Nt
        error("Please check the dimension of source.")
    end
    
    # prepare PML coef
    # case 1: linear
    # pml_value = range(0, stop=pml_coef, length=pml_len);
    # case 2
    pml_value = exp.(-1 ./ range(0, stop=1, length=pml_len).^1)
    pml_value = pml_value ./ maximum(pml_value) * pml_coef
    # case 3
    # pml_value = exp.(-1 ./ range(0, stop=2, length=pml_len).^2)
    # pml_value = pml_value ./ maximum(pml_value) * pml_coef

    Nx_pml = Nx + 2*pml_len;
    Ny_pml = Ny + 2*pml_len;
    
    # setup source and receiver position
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    source_position_pml = source_position .+ pml_len;
    source_position_pml_vec = zeros(Int, source_num)
    if source_num == 1
        source_position_pml_vec[1] = source_position_pml[1] + (source_position_pml[2]-1)*Nx_pml
    else
        for i = 1:source_num
            source_position_pml_vec[i] = source_position_pml[i,1] + (source_position_pml[i,2]-1)*Nx_pml
        end
    end
    receiver_position_pml = receiver_position .+ pml_len;
    receiver_position_pml_vec = zeros(Int, receiver_num)
    if receiver_num == 1
        receiver_position_pml_vec[1] = receiver_position_pml[1] + (receiver_position_pml[2]-1)*Nx_pml
    else
        for i = 1:receiver_num
            receiver_position_pml_vec[i] = receiver_position_pml[i,1] + (receiver_position_pml[i,2]-1)*Nx_pml
        end
    end
    # source integration
    source_int = similar(source)
    @views for i = 1:Nt
        source_int[end-i+1, :] = sum(source[1:end-i+1,:], dims=1) * dt
    end
    
    # Coef
    sigma_x = zeros(Nx_pml, Ny_pml);
    sigma_y = zeros(Nx_pml, Ny_pml);
    @inbounds for i = 1:pml_len
        sigma_x[pml_len+1-i,:] .= pml_value[i];
        sigma_x[pml_len+Nx+i,:] .= pml_value[i];
        sigma_y[:,pml_len+1-i] .= pml_value[i];
        sigma_y[:,pml_len+Ny+i] .= pml_value[i];
    end
    c_ex = extend_vel(c, Nx_pml, Ny_pml, pml_len);
    rho_ex = extend_vel(rho, Nx_pml, Ny_pml, pml_len);
    a = 1 ./ rho_ex;
    b = rho_ex .* c_ex.^2;

    # initialization
    u1 = zeros(Nx_pml, Ny_pml);
    u2 = zeros(Nx_pml, Ny_pml);
    vx1 = zeros(Nx_pml, Ny_pml);
    vx2 = zeros(Nx_pml, Ny_pml);
    vy1 = zeros(Nx_pml, Ny_pml);
    vy2 = zeros(Nx_pml, Ny_pml);
    phi1 = zeros(Nx_pml, Ny_pml);
    phi2 = zeros(Nx_pml, Ny_pml);
    psi1 = zeros(Nx_pml, Ny_pml);
    psi2 = zeros(Nx_pml, Ny_pml);
    U = zeros(Nx, Ny, Nt);
    data = zeros(Nt, receiver_num);

    dxvx1_f = zeros(Nx_pml, Ny_pml);
    dyvy1_f = zeros(Nx_pml, Ny_pml);
    dxvx1_b = zeros(Nx_pml, Ny_pml);
    dyvy1_b = zeros(Nx_pml, Ny_pml);
    dxu2 = zeros(Nx_pml, Ny_pml);
    dyu2 = zeros(Nx_pml, Ny_pml);

    # iter = 1
    u1[source_position_pml_vec] .+= source_int[1] * dt
    @views U[:,:,1] = u1[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
    @views data[1,:] = u1[receiver_position_pml_vec]
    
    # main loop
    @views for iter = 2:Nt

        dx_forward!(dxvx1_f, vx1, dx)
        dy_forward!(dyvy1_f, vy1, dy)
        dx_backward!(dxvx1_b, vx1, dx)
        dy_backward!(dyvy1_b, vy1, dy)

        @. u2 = u1 + dt * (-(sigma_x+sigma_y)*u1 + b*(dxvx1_f+dyvy1_f) + phi1 + psi1)
        @. u2[source_position_pml_vec] .+= b[source_position_pml_vec] .* source_int[iter,:] * dt

        dx_backward!(dxu2, u2, dx)
        dy_backward!(dyu2, u2, dy)

        @. vx2 = vx1 + dt * (a*dxu2 - sigma_x*vx1)
        @. vy2 = vy1 + dt * (a*dyu2 - sigma_y*vy1)
        @. phi2 = phi1 + dt*b*sigma_y*dxvx1_f
        @. psi2 = psi1 + dt*b*sigma_x*dyvy1_f

        copyto!(u1,u2)
        copyto!(vx1,vx2)
        copyto!(vy1,vy2)
        copyto!(phi1,phi2)
        copyto!(psi1,psi2)
        
        @views U[:,:,iter] = u2[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
        @views data[iter,:] = u2[receiver_position_pml_vec]
    end
    return U, data
end

function acoustic_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    
    # reshape
    c = reshape(c, Nx, Ny)
    rho = reshape(rho, Nx, Ny)
    
    if size(source_position,2) != 2
        error("Please check the size of source_position, which should be a Ns times 2 matrix.")
    end
    if size(receiver_position,2) != 2
        error("Please check the size of receiver_position, which should be a Nr times 2 matrix.")
    end
    if size(source,1) != Nt
        error("Please check the dimension of source.")
    end
    
    # prepare PML coef
    # case 1: linear
    # pml_value = range(0, stop=pml_coef, length=pml_len);
    # case 2
    pml_value = exp.(-1 ./ range(0, stop=1, length=pml_len).^1)
    pml_value = pml_value ./ maximum(pml_value) * pml_coef
    # case 3
    # pml_value = exp.(-1 ./ range(0, stop=2, length=pml_len).^2)
    # pml_value = pml_value ./ maximum(pml_value) * pml_coef

    Nx_pml = Nx + 2*pml_len;
    Ny_pml = Ny + 2*pml_len;
    
    # setup source and receiver position
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    source_position_pml = source_position .+ pml_len;
    source_position_pml_vec = zeros(Int, source_num)
    if source_num == 1
        source_position_pml_vec[1] = source_position_pml[1] + (source_position_pml[2]-1)*Nx_pml
    else
        for i = 1:source_num
            source_position_pml_vec[i] = source_position_pml[i,1] + (source_position_pml[i,2]-1)*Nx_pml
        end
    end
    receiver_position_pml = receiver_position .+ pml_len;
    receiver_position_pml_vec = zeros(Int, receiver_num)
    if receiver_num == 1
        receiver_position_pml_vec[1] = receiver_position_pml[1] + (receiver_position_pml[2]-1)*Nx_pml
    else
        for i = 1:receiver_num
            receiver_position_pml_vec[i] = receiver_position_pml[i,1] + (receiver_position_pml[i,2]-1)*Nx_pml
        end
    end
    # source integration
    source_int = similar(source)
    @views for i = 1:Nt
        source_int[end-i+1, :] = sum(source[1:end-i+1,:], dims=1) * dt
    end
    
    # Coef
    sigma_x = zeros(Nx_pml, Ny_pml);
    sigma_y = zeros(Nx_pml, Ny_pml);
    @inbounds for i = 1:pml_len
        sigma_x[pml_len+1-i,:] .= pml_value[i];
        sigma_x[pml_len+Nx+i,:] .= pml_value[i];
        sigma_y[:,pml_len+1-i] .= pml_value[i];
        sigma_y[:,pml_len+Ny+i] .= pml_value[i];
    end
    c_ex = extend_vel(c, Nx_pml, Ny_pml, pml_len);
    rho_ex = extend_vel(rho, Nx_pml, Ny_pml, pml_len);
    a = 1 ./ rho_ex;
    b = rho_ex .* c_ex.^2;

    # initialization
    u1 = zeros(Nx_pml, Ny_pml);
    u2 = zeros(Nx_pml, Ny_pml);
    vx1 = zeros(Nx_pml, Ny_pml);
    vx2 = zeros(Nx_pml, Ny_pml);
    vy1 = zeros(Nx_pml, Ny_pml);
    vy2 = zeros(Nx_pml, Ny_pml);
    phi1 = zeros(Nx_pml, Ny_pml);
    phi2 = zeros(Nx_pml, Ny_pml);
    psi1 = zeros(Nx_pml, Ny_pml);
    psi2 = zeros(Nx_pml, Ny_pml);
    data = zeros(Nt, receiver_num);

    dxvx1_f = zeros(Nx_pml, Ny_pml);
    dyvy1_f = zeros(Nx_pml, Ny_pml);
    dxvx1_b = zeros(Nx_pml, Ny_pml);
    dyvy1_b = zeros(Nx_pml, Ny_pml);
    dxu2 = zeros(Nx_pml, Ny_pml);
    dyu2 = zeros(Nx_pml, Ny_pml);

    # iter = 1
    u1[source_position_pml_vec] .+= source_int[1] * dt
    @views data[1,:] = u1[receiver_position_pml_vec]
    
    # main loop
    @views for iter = 2:Nt

        dx_forward!(dxvx1_f, vx1, dx)
        dy_forward!(dyvy1_f, vy1, dy)
        dx_backward!(dxvx1_b, vx1, dx)
        dy_backward!(dyvy1_b, vy1, dy)

        @. u2 = u1 + dt * (-(sigma_x+sigma_y)*u1 + b*(dxvx1_f+dyvy1_f) + phi1 + psi1)
        @. u2[source_position_pml_vec] .+= b[source_position_pml_vec] .* source_int[iter,:] * dt

        dx_backward!(dxu2, u2, dx)
        dy_backward!(dyu2, u2, dy)

        @. vx2 = vx1 + dt * (a*dxu2 - sigma_x*vx1)
        @. vy2 = vy1 + dt * (a*dyu2 - sigma_y*vy1)
        @. phi2 = phi1 + dt*b*sigma_y*dxvx1_f
        @. psi2 = psi1 + dt*b*sigma_x*dyvy1_f

        copyto!(u1,u2)
        copyto!(vx1,vx2)
        copyto!(vy1,vy2)
        copyto!(phi1,phi2)
        copyto!(psi1,psi2)
        
        @views data[iter,:] = u2[receiver_position_pml_vec]
    end
    return data
end

function acoustic_solver_back(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    if size(source_position,2) != 2
        error("Please check the size of source_position, which should be a Ns times 2 matrix.")
    end
    if size(receiver_position,2) != 2
        error("Please check the size of receiver_position, which should be a Nr times 2 matrix.")
    end
    if size(source,1) != Nt
        error("Please check the dimension of source.")
    end
    
    # prepare PML coef
    # case 1: linear
    # pml_value = range(0, stop=pml_coef, length=pml_len);
    # case 2
    pml_value = exp.(-1 ./ range(0, stop=1, length=pml_len).^1)
    pml_value = pml_value ./ maximum(pml_value) * pml_coef
    # case 3
    # pml_value = exp.(-1 ./ range(0, stop=2, length=pml_len).^2)
    # pml_value = pml_value ./ maximum(pml_value) * pml_coef

    Nx_pml = Nx + 2*pml_len;
    Ny_pml = Ny + 2*pml_len;
    
    # setup source and receiver position
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    source_position_pml = source_position .+ pml_len;
    source_position_pml_vec = zeros(Int, source_num)
    if source_num == 1
        source_position_pml_vec[1] = source_position_pml[1] + (source_position_pml[2]-1)*Nx_pml
    else
        for i = 1:source_num
            source_position_pml_vec[i] = source_position_pml[i,1] + (source_position_pml[i,2]-1)*Nx_pml
        end
    end
    receiver_position_pml = receiver_position .+ pml_len;
    receiver_position_pml_vec = zeros(Int, receiver_num)
    if receiver_num == 1
        receiver_position_pml_vec[1] = receiver_position_pml[1] + (receiver_position_pml[2]-1)*Nx_pml
    else
        for i = 1:receiver_num
            receiver_position_pml_vec[i] = receiver_position_pml[i,1] + (receiver_position_pml[i,2]-1)*Nx_pml
        end
    end
    # source integration
    source_int = similar(source)
    @views for i = 1:Nt
        source_int[end-i+1, :] = sum(source[1:end-i+1,:], dims=1) * dt
    end
    
    # Coef
    sigma_x = zeros(Nx_pml, Ny_pml);
    sigma_y = zeros(Nx_pml, Ny_pml);
    @inbounds for i = 1:pml_len
        sigma_x[pml_len+1-i,:] .= pml_value[i];
        sigma_x[pml_len+Nx+i,:] .= pml_value[i];
        sigma_y[:,pml_len+1-i] .= pml_value[i];
        sigma_y[:,pml_len+Ny+i] .= pml_value[i];
    end
    c_ex = extend_vel(c, Nx_pml, Ny_pml, pml_len);
    rho_ex = extend_vel(rho, Nx_pml, Ny_pml, pml_len);
    a = 1 ./ rho_ex;
    b = rho_ex .* c_ex.^2;

    # initialization
    u1 = zeros(Nx_pml, Ny_pml);
    u2 = zeros(Nx_pml, Ny_pml);
    vx1 = zeros(Nx_pml, Ny_pml);
    vx2 = zeros(Nx_pml, Ny_pml);
    vy1 = zeros(Nx_pml, Ny_pml);
    vy2 = zeros(Nx_pml, Ny_pml);
    phi1 = zeros(Nx_pml, Ny_pml);
    phi2 = zeros(Nx_pml, Ny_pml);
    psi1 = zeros(Nx_pml, Ny_pml);
    psi2 = zeros(Nx_pml, Ny_pml);
    U = zeros(Nx, Ny, Nt);
    data = zeros(Nt, receiver_num);

    dxvx1_f = zeros(Nx_pml, Ny_pml);
    dyvy1_f = zeros(Nx_pml, Ny_pml);
    dxvx1_b = zeros(Nx_pml, Ny_pml);
    dyvy1_b = zeros(Nx_pml, Ny_pml);
    dxu2 = zeros(Nx_pml, Ny_pml);
    dyu2 = zeros(Nx_pml, Ny_pml);

    # iter = 1
    u1[source_position_pml_vec] .+= source_int[1] * dt
    @views U[:,:,1] = u1[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
    @views data[1,:] = u1[receiver_position_pml_vec]
    
    # main loop
    @views for iter = 2:Nt

        dx_forward!(dxvx1_f, vx1, dx)
        dy_forward!(dyvy1_f, vy1, dy)
        dx_backward!(dxvx1_b, vx1, dx)
        dy_backward!(dyvy1_b, vy1, dy)

        @. u2 = u1 + dt * (-(sigma_x+sigma_y)*u1 + b*(dxvx1_f+dyvy1_f) + phi1 + psi1)
        @. u2[source_position_pml_vec] .= b[source_position_pml_vec] .* source_int[iter,:] * dt

        dx_backward!(dxu2, u2, dx)
        dy_backward!(dyu2, u2, dy)

        @. vx2 = vx1 + dt * (a*dxu2 - sigma_x*vx1)
        @. vy2 = vy1 + dt * (a*dyu2 - sigma_y*vy1)
        @. phi2 = phi1 + dt*b*sigma_y*dxvx1_f
        @. psi2 = psi1 + dt*b*sigma_x*dyvy1_f

        copyto!(u1,u2)
        copyto!(vx1,vx2)
        copyto!(vy1,vy2)
        copyto!(phi1,phi2)
        copyto!(psi1,psi2)
        
        @views U[:,:,iter] = u2[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
        @views data[iter,:] = u2[receiver_position_pml_vec]
    end
    return U, data
end
