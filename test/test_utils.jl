function check_grad(g_gpu, g_cpu, atol, rtol; allow_nothing::Bool)
    allow_nothing && return
    @show g_gpu g_cpu
    @test false
end
check_grad(g_gpu::Base.RefValue, g_cpu::Base.RefValue, atol, rtol; allow_nothing::Bool) =
    check_grad(g_gpu[], g_cpu[], atol, rtol; allow_nothing)
check_grad(g_gpu::Nothing, g_cpu::Nothing, atol, rtol; allow_nothing::Bool) =
    @test true
check_grad(g_gpu::Float32, g_cpu::Float32, atol, rtol; allow_nothing::Bool) =
    @test g_cpu ≈ g_gpu rtol=rtol atol=atol
check_grad(g_gpu::CuArray{Float32}, g_cpu::Array{Float32}, atol, rtol; allow_nothing::Bool) =
    @test g_cpu ≈ collect(g_gpu) rtol=rtol atol=atol

function check_grad(g_gpu::Tuple, g_cpu::Tuple, atol, rtol; allow_nothing::Bool)
    for (v1, v2) in zip(g_gpu, g_cpu)
        check_grad(v1, v2, atol, rtol; allow_nothing)
    end
end

function check_grad(g_gpu::NamedTuple, g_cpu::NamedTuple, atol, rtol; allow_nothing::Bool)
    for ((k1,v1), (k2,v2)) in zip(pairs(g_gpu), pairs(g_cpu))
        @test k1 == k2
        check_grad(v1, v2, atol, rtol; allow_nothing)
    end
end

check_type(x) = false
check_type(x::Float32) = true
check_type(x::CuArray{Float32}) = true
check_type(x::Array{Float32}) = true

function gpu_autodiff_test(
    f_cpu, xs_cpu::Array{Float32}...;
    test_equal=true, rtol=1e-4, atol=1e-4,
    checkgrad::Bool = true, allow_nothing::Bool = false,
)
    # Compare CPU & GPU function outputs.
    f_gpu = f_cpu |> gpu
    xs_gpu = gpu.(xs_cpu)

    y_cpu = f_cpu(xs_cpu...)
    y_gpu = f_gpu(xs_gpu...)
    @test collect(y_cpu) ≈ collect(y_gpu) atol=atol rtol=rtol

    checkgrad || return

    ### GRADIENT WITH RESPECT TO INPUT ###

    y_cpu, back_cpu = pullback((x...) -> f_cpu(x...), xs_cpu...)
    @test check_type(y_cpu)
    Δ_cpu = size(y_cpu) == () ? randn(Float32) : randn(Float32, size(y_cpu))
    gs_cpu = back_cpu(Δ_cpu)

    Δ_gpu = Δ_cpu |> gpu
    y_gpu, back_gpu = pullback((x...) -> f_gpu(x...), xs_gpu...)
    @test check_type(y_gpu)
    gs_gpu = back_gpu(Δ_gpu)

    if test_equal
        @test collect(y_cpu) ≈ collect(y_gpu) rtol=rtol atol=atol
        for (g_gpu, g_cpu) in zip(gs_gpu, gs_cpu)
            check_grad(g_gpu, g_cpu, atol, rtol; allow_nothing)
        end
    end

    ### GRADIENT WITH RESPECT TO f ###

    ps_cpu = Flux.params(f_cpu)
    y_cpu, back_cpu = pullback(() -> f_cpu(xs_cpu...), ps_cpu)
    gs_cpu = back_cpu(Δ_cpu)

    ps_gpu = Flux.params(f_gpu)
    y_gpu, back_gpu = pullback(() -> f_gpu(xs_gpu...), ps_gpu)
    gs_gpu = back_gpu(Δ_gpu)

    if test_equal
        @test collect(y_cpu) ≈ collect(y_gpu) rtol=rtol atol=atol
        @assert length(ps_gpu) == length(ps_cpu)
        for (p_gpu, p_cpu) in zip(ps_gpu, ps_cpu)
            check_grad(gs_gpu[p_gpu], gs_cpu[p_cpu], atol, rtol; allow_nothing)
        end
    end
end


test_grad_type(g::Nothing, x) = nothing

function test_grad_type(g::AbstractArray{T1}, x::AbstractArray{T2}) where {T1, T2}
    @test T1 == T2
    @test size(g) == size(x)
end

function test_grad_type(g::NamedTuple, x::T) where T
    for f in fieldnames(T)
        test_grad_type(g[f], getfield(x, f))
    end
end

test_grad_equal(g1::Nothing, g2::Nothing) = nothing

function test_grad_equal(g1::AnyCuArray{T}, g2::Array{T}; atol=1e-4) where T 
    @test Array(g1) ≈ g2 atol=atol
end

function test_grad_equal(g1::T1, g2::T2) where {T1 <: NamedTuple, T2 <: NamedTuple}
    @test fieldnames(T1) == fieldnames(T2)
    for f in fieldnames(T1)
        test_grad_equal(g1[f], g2[f])
    end
end
