using Test

# RED-phase regression test for blocker 2 of the PR #2694 salvage code review.
#
# Contract under test. The helpers live in `test/test_utils_distributed.jl`,
# which does not exist yet, so this file is expected to fail at include time.
# That include failure is the intended RED state; do not implement the helpers
# or modify `test/runtests.jl` in this task.
#
#   strip_distributed_tests!(testsuite) -> testsuite
#       Remove every key that starts with "ext_distributed" from the
#       `Dict{String,Expr}` returned by `ParallelTestRunner.find_tests`.
#       Unrelated keys (e.g. "layers/conv") are left untouched.
#
#   add_distributed_runner!(testsuite, runner_path) -> testsuite
#       Add exactly one key, "ext_distributed", mapped to the expression
#       `:(include($runner_path))`. Child files are never re-added.
#
# Routing contract being pinned:
#   * `test/ext_distributed/runtests.jl` is the SOLE distributed entry point.
#   * Its child directory is ALWAYS excluded from ordinary per-file discovery.
#   * When the distributed flags are enabled, the runner adds a single
#     "ext_distributed" entry that includes the dedicated runner.

include(joinpath(@__DIR__, "test_utils_distributed.jl"))

# A representative discovery result: ordinary tests, an unrelated extension,
# the ext_distributed child files that must never survive discovery, and the
# parent "ext_distributed" entry that is the only one allowed to remain.
function representative_testsuite()
    return Dict{String,Expr}(
        "layers/conv" => :(include("layers/conv.jl")),
        "ext_cuda/cuda" => :(include("ext_cuda/cuda.jl")),
        "ext_distributed/common_distributedtest" =>
            :(include("ext_distributed/common_distributedtest.jl")),
        "ext_distributed/distributed_setup" =>
            :(include("ext_distributed/distributed_setup.jl")),
        "ext_distributed/data_distributedtest" =>
            :(include("ext_distributed/data_distributedtest.jl")),
        "ext_distributed" => :(include("ext_distributed/runtests.jl")),
    )
end

@testset "distributed test routing" begin
    @testset "strip_distributed_tests! removes every ext_distributed key" begin
        suite = representative_testsuite()
        ret = strip_distributed_tests!(suite)

        @test ret === suite
        @test !any(startswith(k, "ext_distributed") for k in keys(suite))
        @test haskey(suite, "layers/conv")
        @test haskey(suite, "ext_cuda/cuda")
        @test !haskey(suite, "ext_distributed")
        @test !haskey(suite, "ext_distributed/common_distributedtest")
        @test !haskey(suite, "ext_distributed/distributed_setup")
        @test !haskey(suite, "ext_distributed/data_distributedtest")
        @test length(suite) == 2
    end

    @testset "add_distributed_runner! adds exactly one ext_distributed entry" begin
        suite = representative_testsuite()
        strip_distributed_tests!(suite)
        runner_path = joinpath(@__DIR__, "ext_distributed", "runtests.jl")
        ret = add_distributed_runner!(suite, runner_path)

        @test ret === suite
        @test haskey(suite, "ext_distributed")
        dist_keys = [k for k in keys(suite) if startswith(k, "ext_distributed")]
        @test dist_keys == ["ext_distributed"]
        @test !any(k -> startswith(k, "ext_distributed/"), keys(suite))
    end

    @testset "added value is an include expression for the runner path" begin
        suite = representative_testsuite()
        strip_distributed_tests!(suite)
        runner_path = joinpath(@__DIR__, "ext_distributed", "runtests.jl")
        add_distributed_runner!(suite, runner_path)

        expr = suite["ext_distributed"]
        @test expr isa Expr
        @test expr.head === :call
        @test expr.args[1] === :include
        @test runner_path in expr.args
        @test occursin(runner_path, string(expr))
    end
end
