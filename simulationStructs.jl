import LinearAlgebra
mutable struct Agent 
	type::Int
	ID::Int
	normNumber::Int
end

function genSingleNorm(num)
	nums = digits(num, base = 2, pad = 4)
	mat = LinearAlgebra.zeros(Int, 2,2)
	mat[1,1] = nums[1]
	mat[2,2] = nums[4]
	mat[2,1] = nums[3]
	mat[1,2] = nums[2]

	# mat = [[nums[4], nums[3]],[nums[2], nums[1]]]
	return mat
end

function genNorm(num)
	ordered = reverse(digits(num, base=16, pad=2))
	return tuple([genSingleNorm(i) for i in ordered]...)
end


# function listDoubler(l)

# 	for i in 1:4
# 		append!(l,l)
# 	end
# 	return l
# end

# const NORMS = [(genNorm(i), genNorm(j)) for i in 0:(16 - 1) for j in 0:(16-1)]

mutable struct EvolutionState 
	population::Array{Agent, 1}	
	reputations::Array{Int8,2}
	statistics::Array{Tuple{Array{Float64,2}, Float64, Array{Int,1}},1}
end