module Optimise

using LinearAlgebra
using Reexport

@reexport using Optimisers

export train!, stop, skip, Schedule
	# ClipValue, ClipNorm,

module Schedule
	using ..Optimise
	using ParameterSchedulers
	import ParameterSchedulers: AbstractSchedule

	include("schedulers.jl")
end

include("optimisers.jl")
include("train.jl")

end
