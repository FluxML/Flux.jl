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

function enzyme_withgradient(f, x...)
    args = []
    for x in x
        if x isa Number
            push!(args, Enzyme.Active(x))
        else
            push!(args, Enzyme.Duplicated(x, Enzyme.make_zero(x)))
        end
    end
    ad = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)
    ret = Enzyme.autodiff(ad, Enzyme.Const(f), Enzyme.Active, args...)
    g = ntuple(i -> x[i] isa Number ? ret[1][i] : args[i].dval, length(x))
    return ret[2], g
end

function _contains_no_numerical(kp, x)
    count = 0
    fmap(x) do y
        if y isa AbstractArray{<:AbstractFloat}
            count += 1
        end
        return y
    end
    return count == 0
end

function check_equal_leaves(a, b; rtol=1e-4, atol=1e-4)
    fmapstructure_with_path(a, b, exclude=_contains_no_numerical) do kp, x, y
        if y isa Nothing
            return
        end
        # @show kp
        if x isa AbstractArray
            @test x ≈ y rtol=rtol atol=atol
        elseif x isa Number
            @test x ≈ y rtol=rtol atol=atol
        end
    end
end

# By default, this computes the gradients on cpu using the default AD (Zygote) 
# and compares them with finite differences.
# Changing the arguments, you can assume the cpu Zygote gradients as the ground truth 
# and test other scenarios.
function test_gradients(
            f, 
            xs...;
            rtol=1e-4, atol=1e-4,
            test_gpu = false,
            test_reactant = false,
            test_enzyme = false,
            test_grad_f = true,
            test_grad_x = true,
            compare_finite_diff = true,
            loss = (f, xs...) -> mean(f(xs...)),
            )

    if !test_gpu && !compare_finite_diff && !test_enzyme && !test_reactant
        error("You should either compare numerical gradients methods or CPU vs GPU.")
    end

    Flux.trainmode!(f) # for layers like BatchNorm

    ## Let's make sure first that the forward pass works.
    l = loss(f, xs...)
    @test l isa Number
    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        xs_gpu = xs |> gpu_dev
        f_gpu = f |> gpu_dev
        l_gpu = loss(f_gpu, xs_gpu...)
        @test l_gpu isa Number
    end

    if test_reactant
        reactant_dev = MLDataDevices.reactant_device(force=true)
        cpu_dev = cpu_device()
        xs_re = xs |> reactant_dev
        f_re = f |> reactant_dev
        l_re = reactant_loss(loss, f_re, xs_re...)
        @test l ≈ l_re rtol=rtol atol=atol
    end

    if test_grad_x
        # Zygote gradient with respect to input.
        y, g = Zygote.withgradient((xs...) -> loss(f, xs...), xs...)
        
        if compare_finite_diff
            # Cast to Float64 to avoid precision issues.
            f64 = f |> Flux.f64
            xs64 = xs .|> Flux.f64
            y_fd, g_fd = finitediff_withgradient((xs...) -> loss(f64, xs...), xs64...)
            @test y ≈ y_fd rtol=rtol atol=atol
            check_equal_leaves(g, g_fd; rtol, atol)
        end

        if test_enzyme
            y_ez, g_ez = enzyme_withgradient((xs...) -> loss(f, xs...), xs...)
            @test y ≈ y_ez rtol=rtol atol=atol
            check_equal_leaves(g, g_ez; rtol, atol)
        end

        if test_gpu
            # Zygote gradient with respect to input on GPU.
            y_gpu, g_gpu = Zygote.withgradient((xs...) -> loss(f_gpu, xs...), xs_gpu...)
            @test get_device(g_gpu) == get_device(xs_gpu)
            @test y_gpu ≈ y rtol=rtol atol=atol
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end

        if test_reactant
            # Enzyme gradient with respect to input on Reactant.
            y_re, g_re = reactant_withgradient(Base.Fix1(loss, f_re), xs_re...)
            @test y ≈ y_re rtol=rtol atol=atol
            check_equal_leaves(g_re |> cpu_dev, g; rtol, atol)
        end
    end

    if test_grad_f
        # Zygote gradient with respect to f.
        y, g = Zygote.withgradient(f -> loss(f, xs...), f)

        if compare_finite_diff
            # Cast to Float64 to avoid precision issues.
            f64 = f |> Flux.f64
            ps, re = Flux.destructure(f64)
            y_fd, g_fd = finitediff_withgradient(ps -> loss(re(ps), xs...), ps)
            g_fd = (re(g_fd[1]),)
            @test y ≈ y_fd rtol=rtol atol=atol
            check_equal_leaves(g, g_fd; rtol, atol)
        end

        if test_enzyme
            y_ez, g_ez = enzyme_withgradient(f -> loss(f, xs...), f)
            @test y ≈ y_ez rtol=rtol atol=atol
            check_equal_leaves(g, g_ez; rtol, atol)
        end

        if test_gpu
            # Zygote gradient with respect to f on GPU.
            y_gpu, g_gpu = Zygote.withgradient(f -> loss(f, xs_gpu...), f_gpu)
            # @test get_device(g_gpu) == get_device(xs_gpu)
            @test y_gpu ≈ y rtol=rtol atol=atol
            check_equal_leaves(g_gpu |> cpu_dev, g; rtol, atol)
        end

        if test_reactant
            # Enzyme gradient with respect to input on Reactant.
            y_re, g_re = reactant_withgradient(Base.Fix2(loss, xs_re[1]), f_re)
            @test y ≈ y_re rtol=rtol atol=atol
            check_equal_leaves(g_re |> cpu_dev, g; rtol, atol)
        end
    end
    return true
end
