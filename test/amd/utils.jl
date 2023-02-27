function amdgputest(
    model, xs...; checkgrad=true, atol=1e-6, allow_nothing::Bool = false,
)
    cpu_model = model
    gpu_model = Flux.gpu(model)

    cpu_in = xs
    gpu_in = Flux.gpu.(xs)

    cpu_out = cpu_model(cpu_in...)
    gpu_out = gpu_model(gpu_in...)
    @test collect(cpu_out) ≈ collect(gpu_out) atol=atol

    if checkgrad
        cpu_grad = gradient(m -> sum(m(cpu_in...)), cpu_model)
        gpu_grad = gradient(m -> sum(m(gpu_in...)), gpu_model)
        amd_check_grad(gpu_grad, cpu_grad; atol, allow_nothing)
    end
end

function amd_check_grad(g_gpu, g_cpu; atol, allow_nothing)
    allow_nothing && return
    @show g_gpu g_cpu
    @test false
end

amd_check_grad(g_gpu::Base.RefValue, g_cpu::Base.RefValue, atol, allow_nothing) =
    amd_check_grad(g_gpu[], g_cpu[]; atol, allow_nothing)
amd_check_grad(g_gpu::Nothing, g_cpu::Nothing; atol, allow_nothing) =
    @test true
amd_check_grad(g_gpu::Float32, g_cpu::Float32; atol, allow_nothing) =
    @test g_cpu ≈ g_gpu atol=atol
amd_check_grad(
    g_gpu::ROCArray{Float32}, g_cpu::Array{Float32};
    atol, allow_nothing,
) = @test g_cpu ≈ collect(g_gpu) atol=atol
amd_check_grad(
    g_gpu::ROCArray{Float32}, g_cpu::Zygote.FillArrays.AbstractFill;
    atol, allow_nothing
) = @test g_cpu ≈ collect(g_gpu) atol=atol

function amd_check_grad(g_gpu::Tuple, g_cpu::Tuple; atol, allow_nothing)
    for (v1, v2) in zip(g_gpu, g_cpu)
        amd_check_grad(v1, v2; atol, allow_nothing)
    end
end

function amd_check_grad(g_gpu::NamedTuple, g_cpu::NamedTuple; atol, allow_nothing)
    for ((k1, v1), (k2, v2)) in zip(pairs(g_gpu), pairs(g_cpu))
        @test k1 == k2
        amd_check_grad(v1, v2; atol, allow_nothing)
    end
end
