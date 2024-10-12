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
    # We set a range to avoid domain errors
    fdm = FiniteDifferences.central_fdm(5, 1, max_range=1e-2)
    return y, FiniteDifferences.grad(fdm, f, x...)
end


function check_equal_leaves(a, b; rtol=1e-4, atol=1e-4)
    fmapstructure_with_path(a, b) do kp, x, y
        if x isa AbstractArray
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
    # Cast to Float64 to avoid precision issues.
    f64 = f |> Flux.f64
    xs64 = xs .|> Flux.f64
    y_fd, g_fd = finitediff_withgradient((xs...) -> loss(f64(xs...)), xs64...)

    # Zygote gradient with respect to input.
    y, g = Zygote.withgradient((xs...) -> loss(f(xs...)), xs...)
    @test y ≈ y_fd rtol=rtol atol=atol
    check_equal_leaves(g, g_fd; rtol, atol)

    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        xs_gpu = xs |> gpu_dev
        f_gpu = f |> gpu_dev

        # Zygote gradient with respect to input on GPU.
        y_gpu, g_gpu = Zygote.withgradient((xs...) -> loss(f_gpu(xs...)), xs_gpu...)
        @test get_device(g_gpu) == gpu_dev
        @test y_gpu |> cpu_dev ≈ y rtol=rtol atol=atol
        check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
    end

    if test_grad_f
        # Use finite differences gradient as a reference.
        # y_fd, g_fd = finitediff_withgradient(f -> loss(f(x)), f)
        ps, re = Flux.destructure(f64)
        y_fd, g_fd = finitediff_withgradient(ps -> loss(re(ps)(xs64...)), ps)
        g_fd = (re(g_fd[1]),)

        # Zygote gradient with respect to f.
        y, g = Zygote.withgradient(f -> loss(f(xs...)), f)
        @test y ≈ y_fd rtol=rtol atol=atol
        check_equal_leaves(g, g_fd; rtol, atol)

        if test_gpu
            # Zygote gradient with respect to input on GPU.
            y_gpu, g_gpu = Zygote.withgradient(f -> loss(f(xs_gpu...)), f_gpu)
            @test get_device(g_gpu) == gpu_dev
            @test y_gpu |> cpu_dev ≈ y rtol=rtol atol=atol
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end
    end
end
