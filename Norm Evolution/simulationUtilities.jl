module simulationUtilities

using LinearAlgebra

export sortPopulation!, calculateGroupProbs, randomSelectGroupWeighted

include("simulationStructs.jl")
using .simulationStructs

function _getWeightedInd(weightedArray, r)
	i = 1
	while r > weightedArray[i]
		i += 1
	end
	return i
end

function calculateCumulativeGroupProbs(groupWeights, groupSizes, groupBounds)
	groupWeights::Array{Float64, 1}
	vals = map(x -> groupWeights[x] * groupSizes[x], 1:length(groupSizes))
	vals = vals ./ sum(vals)
	return [sum(vals[1:i]) for i in 1:length(vals)]
end

function randomSelectGroupWeighted(cumulativeGroupProbs, groupBounds)
	cumulativeGroupProbs::Array{Float64, 1}
	groupBounds::Array{Int, 1}
	g = _getWeightedInd(cumulativeGroupProbs, rand())
	# println("Got group $g")
	return rand((groupBounds[g] + 1):groupBounds[g+1])
end



function sortPopulation!(population, reputations, numGroups)
	# population::Array{Agent,1}
	groupBuckets = [[] for i in 1:numGroups]
	for (i, agent) in enumerate(population)
		append!(groupBuckets[agent.type + 1], [(agent, i)])
	end
	oldReps = copy(reputations)

	j = 1
	# println("groupBuckets length: $(length(groupBuckets))")
	for bucket in groupBuckets
		# println("bucket length is $(length(bucket))")
		# println("first element is $(bucket[1])")
		for (a, i) in bucket
			# println("a: $a i $i")
			reputations[:,j] = oldReps[:,i]
			population[j] = a
			j += 1
		end
	end

	bounds = [0,length(groupBuckets[1])]
	for i in 2:length(groupBuckets)
		append!(bounds, bounds[end] + length(groupBuckets[i]))
	end
	return bounds

end

end