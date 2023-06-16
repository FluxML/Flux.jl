function check_grad(
    g_gpu::ROCArray{Float32}, g_cpu::Array{Float32};
    atol, rtol, allow_nothing::Bool,
)
    @test g_cpu ≈ collect(g_gpu) atol=atol rtol=rtol
end

function check_grad(
    g_gpu::ROCArray{Float32}, g_cpu::Zygote.FillArrays.AbstractFill;
    atol, rtol, allow_nothing::Bool,
)
    @test g_cpu ≈ collect(g_gpu) atol=atol rtol=rtol
end

check_type(x::ROCArray{Float32}) = true
