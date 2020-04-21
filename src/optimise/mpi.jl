function update!(opt, x, x̄)
    if MPI.Initialized()
        x̄′ = zero(x̄)
        MPI.Allreduce!(x̄, x̄′, MPI.SUM, MPI.COMM_WORLD)
        x̄ .= x̄′ ./ MPI.Comm_size(MPI.COMM_WORLD)
    end
    x .-= -apply!(opt, x, x̄)
end

function bcast!(xs)
    if MPI.Initialized()
        for x in xs
            MPI.Bcast!(x, 0, MPI.COMM_WORLD)
        end
    end
end
