# group here all losses, used in tests
const ALL_LOSSES = [Flux.Losses.mse, Flux.Losses.mae, Flux.Losses.msle,
                    Flux.Losses.crossentropy, Flux.Losses.logitcrossentropy,
                    Flux.Losses.binarycrossentropy, Flux.Losses.logitbinarycrossentropy,
                    Flux.Losses.kldivergence,
                    Flux.Losses.huber_loss,
                    Flux.Losses.tversky_loss,
                    Flux.Losses.dice_coeff_loss,
                    Flux.Losses.poisson_loss,
                    Flux.Losses.hinge_loss, Flux.Losses.squared_hinge_loss,
                    Flux.Losses.binary_focal_loss, Flux.Losses.focal_loss,
                    Flux.Losses.siamese_contrastive_loss]


function finitediff_withgradient(f, x...)
    y = f(x...)
    fdm = central_fdm(5, 1)
    return y, FiniteDifferences.grad(fdm, f, x...)
end


function check_equal_leaves(a, b; rtol=1e-4, atol=1e-4, check_eltype=true)
    fmapstructure_with_path(a, b) do kp, x, y
        if x isa AbstractArray
            if check_eltype
                @test eltype(x) == eltype(y)
            end
            @test x ≈ y rtol=rtol atol=atol
        elseif x isa Number
            @test x ≈ y rtol=rtol atol=atol
        end
    end
end


function test_gradients(
            f, 
            xs::Array...;
            rtol=1e-4, atol=1e-4,
            test_gpu = false,
            test_grad_f = true,
            loss = sum
            )

    # Use finite differences gradient as a reference.
    y_fd, g_fd = finitediff_withgradient((xs...) -> loss(f(xs...)), xs...)

    # Zygote gradient with respect to input.
    y, g = Zygote.withgradient((xs...) -> loss(f(xs...)), xs...)
    @test y ≈ y_fd rtol=rtol atol=atol
    check_equal_leaves(g, g_fd; rtol, atol)

    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        x_gpu = x |> gpu_dev
        f_gpu = f |> gpu_dev

        # Zygote gradient with respect to input on GPU.
        y_gpu, g_gpu = Zygote.withgradient(x -> loss(f_gpu(x)), x_gpu)
        @test get_device(g_gpu) == gpu_dev
        @test y_gpu |> cpu_dev ≈ y rtol=rtol atol=atol
        check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
    end

    if test_grad_f
        # Use finite differences gradient as a reference.
        # y_fd, g_fd = finitediff_withgradient(f -> loss(f(x)), f)
        ps, re = Flux.destructure(f)
        y_fd, g_fd = finitediff_withgradient(f -> loss(re(ps)(x)), ps)
        g_fd = (re(g_fd[1]),)

        # Zygote gradient with respect to f.
        y, g = Zygote.withgradient(f -> loss(f(x)), f)
        @test y ≈ y_fd rtol=rtol atol=atol
        check_equal_leaves(g, g_fd; rtol, atol)

        if test_gpu
            # Zygote gradient with respect to input on GPU.
            y_gpu, g_gpu = Zygote.withgradient(f -> loss(f(x_gpu)), f_gpu)
            @test get_device(g_gpu) == gpu_dev
            @test y_gpu |> cpu_dev ≈ y rtol=rtol atol=atol
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end
    end
end

# check_grad_type checks that the gradient type matches the primal type.

check_grad_type(g::Nothing, x) = nothing

function check_grad_type(g::AbstractArray{T1}, x::AbstractArray{T2}) where {T1, T2}
    @test T1 == T2
    @test size(g) == size(x)
end

function check_grad_type(g::NamedTuple, x::T) where T
    for f in fieldnames(T)
        check_grad_type(g[f], getfield(x, f))
    end
end
