using Pkg
Pkg.activate(@__DIR__)      # no need to instantiate here

using JuliaFormatter

help = """
Usage: flux_format.jl [flags] [FILE/PATH]...

Formats the given julia files using the Flux formatting options.
If paths are given instead, it will format all *.jl files under
the paths. If nothing is given, all changed julia files are formatted.

    -v, --verbose
        Print the name of the files being formatted with relevant details.

    -h, --help
        Print this help message.
"""

options = Dict{Symbol, Bool}()
indices_to_remove = []      # used to delete options once processed

for (index, arg) in enumerate(ARGS)
    if arg[1] != '-'
        continue
    end
    if arg in ["-v", "--verbose"]
        opt = :verbose
        push!(indices_to_remove, index)
    elseif arg in ["-h", "--help"] 
        opt = :help 
        push!(indices_to_remove)
    else
        error("Option $arg is not supported.")
    end
    options[opt] = true
end

# remove options from args
deleteat!(ARGS, indices_to_remove)

# print help message if asked
if haskey(options, :help)
    write(stdout, help)
    exit(0)
end

# otherwise format files
if isempty(ARGS)
    filenames = readlines(`git ls-files "*.jl"`)
else
    filenames = ARGS
end

format(filenames; options...)
