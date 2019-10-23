module simulationAgentInteractions

export moveError, updateReps!, batchUpdate!

include("simulationStructs.jl")
include("simulationUtilities.jl")
using .simulationStructs
using .simulationUtilities

function moveError(move, ec)
	if move == 0
		return 0
	elseif rand() < ec
		return 0
	else 
		return 1
	end
end

function updateReps!(reputations, population, a,b,action, perpetratorNorms, relativeNorms, uvisibility, NORMS)
	if relativeNorms
		normInd = 1
		if rand() < uvisibility
			if population[a].type != population[b].type
				normInd = 2
			end
		end
	elseif ! perpetratorNorms
		normInd = population[b].type + 1 #COOL OPTIONS HERE
	else
		normInd = population[a].type + 1
	end

	for j in 1:length(NORMS)
		jsview = reputations[j,b]
		newrep = NORMS[j][normInd][action + 1, jsview + 1]
		reputations[j,a] = newrep
	end
end



function batchUpdate!(simparams, reputations, population, groupBounds, roundPayoffs, cooperationRate, NORMS, egalitarianAgentInteractions)
	numagents::Int = length(NORMS) * simparams["NUMAGENTSPERNORM"]
	batchsize::Int = simparams["BATCHSIZE"]

	gameBenefit::Float64 = simparams["gameBenefit"]
	gameCost::Float64 = simparams["gameCost"]
		
	oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]

	
	perpetratorNorms::Bool = simparams["perpetratorNorms"]
	relativeNorms::Bool = simparams["relativeNorms"]
	uvisibility::Float64 = simparams["uvisibility"]

	a = trunc(Int, ceil(rand() * numagents))
	normNumber = population[a].normNumber
	b = 0
	action = 0

	if ! egalitarianAgentInteractions

		groupBounds::Array{Int,1}
		groupSizes = [groupBounds[i] - groupBounds[i-1] for i in 2:length(groupBounds)]
		# println("groupSizes: $groupSizes")

		cumGroupProbs = calculateCumulativeGroupProbs(simparams["groupWeights"],  groupSizes, groupBounds)
		# println("groupProbs: $groupProbs")
	end


	for j in 1:batchsize
		if ! egalitarianAgentInteractions
			b = randomSelectGroupWeighted(cumGroupProbs, groupBounds)
		else
			b = trunc(Int, ceil(rand() * numagents))
		end
		#adversary
		action = reputations[normNumber, b]
		# if !relativeNorms
		# 	action = reputations[normNumber, b]
		# elseif atype
		# 	action = reputations[mirroredNormNumber,b]
		# else
		# 	action = reputations[normNumber, b]
		# action = moveError(action, Ecoop)

		aPayoff, bPayoff = oneShotMatrix[action + 1]

		roundPayoffs[a] += aPayoff
		roundPayoffs[b] += bPayoff

		if a == b
			
			updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility, NORMS)
		end

		cooperationRate += action

	end

	updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility, NORMS)

	return cooperationRate
end


end
