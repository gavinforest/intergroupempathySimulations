module simulationUpdateRules

export imitationUpdate!, pairwiseComparison!, mutatePopulation!, deathBirthUpdate!, birthDeathUpdate!

include("simulationUtilities.jl")
using .simulationUtilities
import StatsBase
import LinearAlgebra

function getAgentPair(numAgents, intergroupUpdateP, groupBounds, population)
	ind1 = trunc(Int,ceil(rand() * numAgents)) 
	ind2 = trunc(Int, ceil(rand() * numAgents)) 

	inter = rand() < intergroupUpdateP
	
	ind1type = population[ind1].type
	ind2type = population[ind2].type 
	    
	if ind2type != ind1type && inter
		ind2 = rand((groupBounds[ind1type]+1):groupBounds[ind1type + 1])
		ind2type = ind1type
	end

	return ind1,ind2
end

function copyAgent!(ind1, ind2, population, imitationCoupling, typeMigrate, groupBounds)
	if rand() < imitationCoupling
		population[ind1].normNumber = population[ind2].normNumber
	else
		if rand() < 0.5
			newNum = (population[ind1].normNumber & 15) + (population[ind2].normNumber & (255 - 15))
		else
			newNum = (population[ind2].normNumber & 15) + (population[ind1].normNumber & (255 - 15))
		end

		population[ind1].normNumber = newNum
	end

	ind1type = population[ind1].type 
	ind2type = population[ind2].type


	if typeMigrate
		# groupSets.memberSets[ind1type+1] = setdiff(groupSets.memberSets[ind1type+1], BitSet(ind1))
		# groupSets.sizes[ind1type+1] -= 1
		population[ind1].type = ind2type
		# groupSets.memberSets[ind2type] = union(groupSets.memberSets[ind2type], BitSet(ind1))
		# groupSets.sizes[ind2type] += 1
		
	end
end

function pairwiseComparison!(population, roundPayoffs, groupBounds, numImitate, simParams)
	intergroupUpdateP::Float64 = simParams["intergroupUpdateP"]
	typeMigrate::Bool = simParams["typeImitate"]
	w::Float64= simParams["w"]
	imitationCoupling::Float64= simParams["imitationCoupling"]



	changed = Set()
	numAgents = length(population)


	for i in 1:numImitate
		ind1,ind2 = getAgentPair(numAgents, intergroupUpdateP, groupBounds, population)

		if ind1 != ind2 && ! in(ind1, changed) && ! in(ind2, changed)

			pCopy = 1.0 / (1.0 + exp( (- w) * (roundPayoffs[ind2] - roundPayoffs[ind1])))
			
			if rand() < pCopy

				copyAgent!(ind1, ind2, population, imitationCoupling, typeMigrate, groupBounds)

				# population[ind1].normNumber = population[ind2].normNumber
				push!(changed, ind1)
			end
		end
	end
end

function imitationUpdate!(population, roundPayoffs, numImitate, simParams)
	w::Float64 = simParams["w"]

	numAgents = length(population)

	fermiPayoffWeights = StatsBase.Weights(map(x -> exp( w * x), roundPayoffs))
	uniformWeights = StatsBase.Weights(LinearAlgebra.ones(Float64, numAgents) / numAgents)

	killed = StatsBase.sample(1:numAgents, uniformWeights, numImitate, replace=false)
	replacements = StatsBase.sample(1:numAgents, fermiPayoffWeights , numImitate)

	# updates = []
	updates = map(i -> (killed[i], population[replacements[i]]), 1:numImitate)
	# for i in 1:numImitate
	# 	append!(updates, (killed[i], population[replacements[i]]))
	# end

	for update in updates
		target, agent = update
		population[target] = agent
	end
end

function deathBirthUpdate!(population, roundPayoffs, numImitate, simParams)
	w::Float64 = simParams["w"]

	numAgents = length(population)

	fermiPayoffWeights = StatsBase.Weights(map(x -> exp( w * x), roundPayoffs))
	uniformWeights = StatsBase.Weights(LinearAlgebra.ones(Float64, numAgents) / numAgents)

	killed = StatsBase.sample(1:numAgents, uniformWeights, numImitate, replace=false)
	replacements = StatsBase.sample(1:numAgents, fermiPayoffWeights , numImitate)

	updates = []

	for i in 1:numAgents
		if ! in(replacements[i], killed)
			append!(updates, (killed[i], population[replacements[i]]))
		end
	end

	for update in updates
		target, agent = update
		population[target] = agent
	end
end


function birthDeathUpdate!(population, roundPayoffs, numImitate, simParams)
	w::Float64 = simParams["w"]

	numAgents = length(population)

	fermiPayoffWeights = StatsBase.Weights(map(x -> exp( w * x), roundPayoffs))
	uniformWeights = StatsBase.Weights(LinearAlgebra.ones(Float64, numAgents) / numAgents)

	killed = StatsBase.sample(1:numAgents, uniformWeights, numImitate, replace=false)
	replacements = StatsBase.sample(1:numAgents, fermiPayoffWeights , numImitate)

	updates = []

	for i in 1:numAgents
		if ! in(killed[i], replacements)
			append!(updates, (killed[i], population[replacements[i]]))
		end
	end

	for update in updates
		target, agent = update
		population[target] = agent
	end
end


# function calculateGroupSets(population, numGroups)
# 	groupSets = [BitSet() for i in 1:numGroups]
# 	for i in 1:length(population)
# 		union!(groupSets[population[i].type + 1], BitSet(i))
# 	end
# 	sizes = map(length, groupSets)
# 	return Groups(sizes, groupSets)
# end

function mutatePopulation!(population, ustrat, NORMS)
	#random drift applied uniformly to all individuals
	for j in 1:length(population)
		if rand() < ustrat
			num = trunc(Int, ceil(rand() * length(NORMS)))
			population[j].normNumber = num
		end

		# if population[j].type == 0 && u01 > rand()
		# 	population[j].type = 1
		# elseif population[j].type == 1 && u10 > rand()
		# 	population[j].type = 0
		# end
	end

end

end
