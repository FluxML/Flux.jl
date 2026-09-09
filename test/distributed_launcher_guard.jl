using Test
using Flux: DistributedUtils

# Contract tests for `DistributedUtils.check_launcher_compat`, the pure launcher
# compatibility checker that backs the MPI-backend PMI guardrail in
# `ext/FluxMPIExt/FluxMPIExt.jl`.
#
# The checker must be a pure decision function: it receives the environment as
# a plain dictionary and the *name* of the loaded MPI library (as reported by
# `MPI.MPI_LIBRARY`, e.g. "MPICH", "OpenMPI", "MPItrampoline", "unknown"), plus
# the two guard gates `force` and `mpi_initialized`. It must NOT initialize or
# even load MPI, so the full decision matrix is testable without a launcher.
#
# Contract:
#   check_launcher_compat(env; library, force=false, mpi_initialized=false)
#     -> `nothing` when the launch environment is compatible with the loaded
#        library (no guard action required), or
#     -> a non-empty `String` error message when a KNOWN unsafe mismatch is
#        detected. The message must name the loaded library, name the launcher
#        variable that exposed the mismatch, and mention the `force=true`
#        bypass.
#
# Decision matrix (see temp/pr_salvage_plan.md, phase 3):
#   library | environment                    | force | mpi_initialized | result
#   --------|--------------------------------|-------|-----------------|-------
#   MPICH   | PMIX_RANK                      | false | false           | error
#   MPICH   | OMPI_COMM_WORLD_RANK           | false | false           | error
#   MPICH   | PMIX_RANK and/or OMPI vars     | true  | false           | ok
#   MPICH   | PMIX_RANK and/or OMPI vars     | false | true            | ok
#   OpenMPI | PMIx/OpenMPI launcher vars     | false | false           | ok
#   any     | no PMIx/OpenMPI launcher vars  | false | false           | ok
#
# Extra rows pin the "detect only the known unsafe MPICH case" requirement:
# non-MPICH non-OpenMPI libraries are never judged, and the presence of
# SLURM/PMI2 variables never suppresses a conclusive PMIx/OpenMPI mismatch.

function launch_env(; pmix::Bool=false, ompi::Bool=false, pmi2::Bool=false,
        slurm::Bool=false, value::AbstractString="0")
    env = Dict{String,String}()
    pmix  && (env["PMIX_RANK"] = value)
    ompi  && (env["OMPI_COMM_WORLD_RANK"] = value)
    pmi2  && (env["PMI2_RANK"] = value)
    slurm && (env["SLURM_JOB_ID"] = "12345")
    return env
end

@testset "check_launcher_compat decision matrix" begin
    @testset "known unsafe: MPICH under a PMIx launcher environment" begin
        for (name, env) in [
                ("PMIX_RANK present", launch_env(pmix=true)),
                ("PMIX_RANK present with empty value", launch_env(pmix=true, value="")),
                ("PMIX_RANK inside a SLURM job", launch_env(pmix=true, slurm=true)),
                ("PMIX_RANK + OMPI_COMM_WORLD_RANK", launch_env(pmix=true, ompi=true)),
                ("PMIX_RANK + OMPI + SLURM", launch_env(pmix=true, ompi=true, slurm=true))]
            r = DistributedUtils.check_launcher_compat(
                env; library="MPICH", force=false, mpi_initialized=false)
            @test r isa String
            @test !isempty(r)
            @test occursin("MPICH", r)
            @test occursin("PMIX_RANK", r)
            @test occursin("force=true", r)
        end
    end

    @testset "known unsafe: MPICH launched by OpenMPI mpirun" begin
        for (name, env) in [
                ("OMPI_COMM_WORLD_RANK present", launch_env(ompi=true)),
                ("OMPI_COMM_WORLD_RANK with empty value", launch_env(ompi=true, value="")),
                ("OMPI_COMM_WORLD_RANK inside a SLURM job", launch_env(ompi=true, slurm=true))]
            r = DistributedUtils.check_launcher_compat(
                env; library="MPICH", force=false, mpi_initialized=false)
            @test r isa String
            @test !isempty(r)
            @test occursin("MPICH", r)
            @test occursin("OMPI_COMM_WORLD_RANK", r)
            @test occursin("force=true", r)
        end
    end

    @testset "force=true bypasses every known mismatch" begin
        for env in (launch_env(pmix=true), launch_env(ompi=true),
                launch_env(pmix=true, ompi=true), launch_env(pmix=true, slurm=true))
            @test DistributedUtils.check_launcher_compat(
                env; library="MPICH", force=true, mpi_initialized=false) === nothing
        end
    end

    @testset "guard never runs when MPI is already initialized" begin
        # Even the known unsafe MPICH/PMIx combination must pass once MPI.Init
        # has happened: the guard runs only before MPI.Init().
        for library in ("MPICH", "OpenMPI", "MPItrampoline", "unknown")
            for env in (launch_env(pmix=true), launch_env(ompi=true),
                    launch_env(pmix=true, ompi=true))
                @test DistributedUtils.check_launcher_compat(
                    env; library=library, force=false, mpi_initialized=true) === nothing
            end
        end
    end

    @testset "system OpenMPI is always allowed" begin
        for env in (launch_env(), launch_env(pmix=true), launch_env(ompi=true),
                launch_env(pmix=true, ompi=true), launch_env(pmix=true, slurm=true),
                launch_env(pmi2=true, slurm=true))
            @test DistributedUtils.check_launcher_compat(
                env; library="OpenMPI", force=false, mpi_initialized=false) === nothing
        end
    end

    @testset "no PMIx/OpenMPI launcher variables is always ok (incl. PMI2/SLURM envs)" begin
        # A real MPICH_jll job launched by a PMI2 launcher carries PMI2/SLURM
        # variables, never PMIX_RANK/OMPI_COMM_WORLD_RANK. These must pass.
        for env in (launch_env(), launch_env(pmi2=true), launch_env(slurm=true),
                launch_env(pmi2=true, slurm=true), Dict{String,String}("PATH" => "/usr/bin"))
            @test DistributedUtils.check_launcher_compat(
                env; library="MPICH", force=false, mpi_initialized=false) === nothing
        end
        # A PMI1-era variable is not a conclusive mismatch indicator either.
        @test DistributedUtils.check_launcher_compat(
            Dict{String,String}("PMI_RANK" => "0");
            library="MPICH", force=false, mpi_initialized=false) === nothing
    end

    @testset "detect only the known unsafe MPICH case (no broad claims)" begin
        # Non-MPICH, non-OpenMPI libraries must never be judged: the guard has
        # no conclusive knowledge about their PMI requirements.
        for library in ("MPItrampoline", "MicrosoftMPI", "IBMSpectrumMPI",
                "IntelMPI", "MVAPICH", "FujitsuMPI", "unknown")
            for env in (launch_env(pmix=true), launch_env(ompi=true),
                    launch_env(pmix=true, ompi=true))
                @test DistributedUtils.check_launcher_compat(
                    env; library=library, force=false, mpi_initialized=false) === nothing
            end
        end
    end

    @testset "result contract: nothing or non-empty String, never throws" begin
        libraries = ("MPICH", "OpenMPI", "MPItrampoline", "MicrosoftMPI", "unknown")
        envs = (launch_env(), launch_env(pmix=true), launch_env(ompi=true),
                launch_env(pmi2=true, slurm=true))
        for library in libraries, env in envs, force in (false, true),
                mpi_initialized in (false, true)
            r = DistributedUtils.check_launcher_compat(
                env; library=library, force=force, mpi_initialized=mpi_initialized)
            @test r === nothing || (r isa String && !isempty(r))
        end
    end
end
