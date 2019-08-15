module simulatioinUtilities

using LinearAlgebra

export sortPopulation!, calculateGroupProbs, randomSelectGroupWeighted



function _getWeightedInd(weightedArray, r)
	s = 0.0
	i = 0
	while s < r
		i += 1
		s += weightedArray[i]
	end
	return i
end

function calculateGroupProbs(groupWeights, groupSizes, groupBounds)
	groupweights::Array{Float, 1}
	groupSets::Groups
	totalProb = sum(x -> groupweights[x] * groupSizes[x], 1:length(groupSizes))
	return map(x -> groupweights[x] * groupSizes[x] / totalProb, 1:length(groupSizes))
end

function randomSelectGroupWeighted(groupProbs, bounds)
	groupProbs::Array{Float, 1}
	g = _getWeightedInd(groupProbs, rand())
	return rand((bounds[g -1] + 1):bounds[g])
end



function sortPopulation!(population, reputations, numGroups)
	groupBuckets = [[] for i in 1:numGroups] :: Array{Array{Tuple{Agent, Int},1},1}
	for i, agent in enumerate(population)
		append!(groupBuckets[agent.type + 1], (agent, i))
	end
	oldReps = copy(reputations)

	j = 1
	for bucket in groupBuckets
		for a, i in bucket
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