check_grad(g_gpu::CuArray{Float32}, g_cpu::Array{Float32}; rtol=1e-4, atol=1e-4, allow_nothing::Bool=false) =
    @test g_cpu ≈ collect(g_gpu) rtol=rtol atol=atol

check_type(x::CuArray{Float32}) = true
