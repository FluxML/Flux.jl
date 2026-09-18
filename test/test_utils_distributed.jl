# Routing helpers for the distributed test suite, kept out of `runtests.jl` so
# that `test/distributed_routing.jl` can unit-test them. `runtests.jl` removes
# this file from ParallelTestRunner discovery (it is not a test).

const DISTRIBUTED_TEST_ENTRYPOINT = "ext_distributed"

"""
    strip_distributed_tests!(testsuite)

Remove every `ext_distributed` entry (the dedicated runner and its child test
files) from the discovered `testsuite`. The dedicated runner is re-added by
[`add_distributed_runner!`](@ref) only when distributed tests are enabled.
"""
function strip_distributed_tests!(testsuite)
    filter!(p -> !startswith(first(p), DISTRIBUTED_TEST_ENTRYPOINT), testsuite)
    return testsuite
end

"""
    add_distributed_runner!(testsuite, runner_path)

Add the single `"ext_distributed"` entry that includes the dedicated runner at
`runner_path`. Child test files are never re-added.
"""
function add_distributed_runner!(testsuite, runner_path)
    testsuite[DISTRIBUTED_TEST_ENTRYPOINT] = :(include($runner_path))
    return testsuite
end
