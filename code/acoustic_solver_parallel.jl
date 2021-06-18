using Distributed, SharedArrays

@everywhere include("code/acoustic_solver.jl")

# forward modelling operator
function multi_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=200)
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    data = SharedArray{Float64}(Nt, receiver_num, source_num)
    U = SharedArray{Float64}(Nx, Ny, Nt, source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        U1, data1 = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        data[:,:,ind] = data1;
        U[:,:,:,ind] = U1;
    end
    data = Array(data)
    U = Array(U)
    
    return U, data
end

function multi_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=200)
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    data = SharedArray{Float64}(Nt, receiver_num, source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        data[:,:,ind] = acoustic_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    end
    data = Array(data)
    
    return data
end

