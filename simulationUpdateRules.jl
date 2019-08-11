function getAgentPair(numAgents, intergroupUpdateP, groupSets, population)
	ind1 = trunc(Int,ceil(rand() * numAgents)) 
	ind2 = trunc(Int, ceil(rand() * numAgents)) 

	inter = rand() < intergroupUpdateP
	
	ind1type = population[ind1].type
	ind2type = population[ind2].type 
	    
	if ind2type != ind1type && inter
		ind2 = rand(groupSets[ind1type])
		ind2type = ind1type
	end

	return ind1,ind2
end

function copyAgent!(ind1, ind2, population, imitationCoupling, typeMigrate, groupSets)
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

	if typeMigrate
		groupSets[ind1type] = setdiff(groupSets[ind1type], BitSet(ind1))
		population[ind1].type = ind2type
		groupSets[ind2type] = union(groupSets[ind2type], BitSet(ind1))
		
	end
end

function imitationUpdate!(population, roundPayoffs, groupSets, numImitate, intergroupUpdateP, typeMigrate,w, imitationCoupling)
	changed = Set()
	numAgents = length(population)


	for i in 1:numImitate
		ind1,ind2 = getAgentPair(numAgents, intergroupUpdateP, groupSets, population)

		if ind1 != ind2 && ! in(ind1, changed) && ! in(ind2, changed)

			pCopy = 1.0 / (1.0 + exp( (- w) * (roundPayoffs[ind2] - roundPayoffs[ind1])))
			
			if rand() < pCopy

				copyAgent!(ind1, ind2, population, imitationCoupling, typeMigrate, groupSets)

				# population[ind1].normNumber = population[ind2].normNumber
				push!(changed, ind1)
			end
		end
	end
end

function deathBirthUpdate(population, roundPayoffs, numImitate, w)
	numAgents = length(population)

	payoffWeights = StatsBase.Weights(roundPayoffs)
	uniformWeights = StatsBase.Weights(LinearAlgebra.ones(Float64, numAgents) / numAgents)

	killed = StatsBase.sample(1:numAgents, uniformWeights, numImitate, replace=false)
	replacements = StatsBase.sample(1:numAgents, payoffWeights, numImitate)

	updates = []

	for i in 1:numAgents

		append!(updates, (killed[i], population[replacements[i]]))
	end

	for update in updates
		target, agent = update
		population[target] = agent
	end
end

function calculateGroupSets(population, numGroups)
	groupSets = [BitSet() for i in 1:numGroups]
	for i in 1:length(population)
		union!(groupSets[population[i].type + 1], BitSet(i))
	end
	return groupSets
end

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