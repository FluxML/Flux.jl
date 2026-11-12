module FluxNVTXExt

using Flux: Flux
using NVTX: NVTX

# Made at load time: a domain made while precompiling would keep a pointer from that process.
const DOMAIN = Ref{NVTX.Domain}()

function Flux.Train._range_start(name::String)
    NVTX.isactive() || return nothing
    return NVTX.range_start(NVTX.init!(DOMAIN[]); message = name, color = hash(name) % UInt32)
end

Flux.Train._range_end(id::NVTX.RangeId) = NVTX.range_end(id)

function __init__()
    DOMAIN[] = NVTX.Domain("Flux")
    Flux.Train.RANGES[] = true
end

end
