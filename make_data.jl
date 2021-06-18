using JLD2, LinearAlgebra

include("code/acoustic_solver_parallel.jl")
include("code/adjoint_method.jl")

# load data
@load "marmousi_model/marmousi_model_for_ex4.jld2"
# The old one is right
# @load "OT_work/Ex_4/ex4_data.jld2"

Nx, Ny = size(c_true)
rho_true = copy(rho)

# time
Fs = 400;
dt = 1/Fs;
Nt = 1200;
t = range(0,length=Nt,step=dt);

# source
source = source_ricker(5,0.2,t)
source_num = 11
source_position = zeros(Int,source_num,2)
for i = 1:source_num
        source_position[i,:] = [1 1+30(i-1)]
end
source = repeat(source, 1, 1);

# receiver
receiver_num = 101
receiver_position = zeros(Int,receiver_num,2)
for i = 1:receiver_num
    receiver_position[i,:] = [1, (i-1)*3+1]
end

# PML
pml_len = 30
pml_coef = 50

println("Nx: ", Nx, ". Ny: ", Ny, ". Nt:", Nt)
println("Source number: ", source_num)
println("Receiver number: ", receiver_num)
println("CFL: ", maximum(c_true) * dt / h);
println("===============================================")

# make data
println("Computing received data.")
@time received_data = multi_solver_no_wavefield(c_true, rho_true, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
println("Received data done.")

println("Saving.")
@save "temp_data/data.jld2" Nx Ny h c c_true rho rho_true Fs dt Nt t source source_num source_position receiver_num receiver_position pml_len pml_coef received_data
println("Saved.")
