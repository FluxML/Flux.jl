import MPI
MPI.Init()

comm = MPI.COMM_WORLD
N = 5
root = 0

if MPI.Comm_rank(comm) == root
    print(" Running on $(MPI.Comm_size(comm)) processes\n")
end
MPI.Barrier(comm)

if MPI.Comm_rank(comm) == root
    A = [i*(1.0 + im*2.0) for i = 1:N]
else
    A = Array{ComplexF64}(undef, N)
end

MPI.Bcast!(A, root, comm)

print("rank = $(MPI.Comm_rank(comm)), A = $A\n")

if MPI.Comm_rank(comm) == root
    B = Dict("foo" => "bar")
else
    B = nothing
end

B = MPI.bcast(B, root, comm)
print("rank = $(MPI.Comm_rank(comm)), B = $B\n")

if MPI.Comm_rank(comm) == root
    f = x -> x^2 + 2x - 1
else
    f = nothing
end

f = MPI.bcast(f, root, comm)
print("rank = $(MPI.Comm_rank(comm)), f(3) = $(f(3))\n")