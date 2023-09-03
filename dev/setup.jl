# instantiate the environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# setup the custom git hook
using Git

# set the local hooks path
const git = Git.git()
run(`$git config --local core.hooksPath .githooks/`)

# set file permission for hook
Base.Filesystem.chmod(".githooks", 0o777; recursive = true)
