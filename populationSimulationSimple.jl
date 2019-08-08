using Distributed
import LinearAlgebra
import StatsBase


const DEBUG = true
const PROGRESSVERBOSE = true
const TEST = false



# const SS = [[1, 0], [1, 1]]
# const SJ = [[1, 0], [0, 1]]
# const SC = [[0, 0], [1, 1]]
# const SH = [[0, 0], [0, 1]]

# const NORMS = [SC, SH, SJ, SS]
# const STRATNUMS = collect(1:4)
# const STRATNAMES = ["SC", "SH", "SJ", "SS"]



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
const NORMS = [genNorm(i) for i in 0:(16^2 -1)]
# const NORMS = listDoubler([genNorm(i) for i in 0:(16 - 1)])
# for i in 1:4
# 	append!(NORMS, NORMS)
# end




# function makeAgent(type, ID, normNumber)
# 	# numString = base(4,normNumber, 4)
# 	# norms = [NORMS[parse(Int,s)] for s in numString]
# 	myAgent = Agent(type, ID, normNumber)
# 	return myAgent
# end

function moveError(move, ec)
	if move == 0
		return 0
	elseif rand() < ec
		return 0
	else 
		return 1
	end
end

function mirrorNormNumber(num)
	return ((num >> 4) + (num<<4))&255
end



# function typeStratDoubleToNum(firstTup, secondTup)
# 	(firstType, firstStratNumber) = firstTup
# 	(secondType, secondStratNumber) = secondTup
# 	return 6 * typeStratToNum((firstType, firstStratNumber)) + typeStratToNum((secondType, secondStratNumber))
# end

function generateStatistics!(statList,generation, population, numGroups, cooperationRate)
	normProportions = LinearAlgebra.zeros(Float64, length(NORMS), numGroups)
	# type1prop = 0
	typeNums = [0 for i in 1:numGroups]
	typeNums::Array{Int,1}

	for j in 1:length(population)
		normProportions[population[j].normNumber, population[j].type + 1] += 1
		typeNums[population[j].type + 1] += 1

	end

	# type1prop = type1prop / length(population)

	normProportions = normProportions / length(population)

	statList[generation] = (normProportions, cooperationRate, typeNums)
	# statList[generation] = (normProportions, cooperationRate, 1.0 - type1prop, type1prop)
	return statList

end

function imageMatrix(reputations, population)

	numNorms = length(NORMS)
	numAgents = size(reputations)[2] #assumption about indexing here

	imageMatrix = LinearAlgebra.zeros(Float64, numNorms, numNorms)

	for i in 1:numNorms
		reps = LinearAlgebra.zeros(Float64, numNorms, 2)
		for n in 1:length(population)

			reps[population[n].normNumber,1] += reputations[i,n]
			reps[population[n].normNumber,2] += 1.0

		end
		reps[:,1] = reps[:,1] ./ reps[:,2]
		imageMatrix[i,:] = reps[:,1]
	end
	return imageMatrix
end


function updateReps!(reputations, population, a,b,action, perpetratorNorms, relativeNorms, uvisibility)
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

function batchUpdate!(numagents,batchsize, reputations, population, perpetratorNorms, relativeNorms, uvisibility, oneShotMatrix, roundPayoffs, cooperationRate)
	a = trunc(Int, ceil(rand() * numagents))
	normNumber = population[a].normNumber
	b = 0
	action = 0

	for j in 1:batchsize
		b = trunc(Int, ceil(rand() * numagents))
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
			
			updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility)
		end

		cooperationRate += action

	end

	updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility)

	return cooperationRate
end

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

mutable struct EvolutionState 
	population::Array{Agent, 1}	
	reputations::Array{Int8,2}
	statistics::Array{Tuple{Array{Float64,2}, Float64, Array{Int,1}},1}
end

function calculateGroupSets(population, numGroups)
	groupSets = tuple([BitSet() for i in 1:numGroups]...)
	for i in 1:length(population)
		union!(groupSets[population[i].type + 1], BitSet(i))
	end
	return groupSets
end

function mutatePopulation!(population, ustrat)
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




println("Number of norms: $(length(NORMS))")
println("Number of processes: $(nprocs())")

function evolve(parameterDictionary, state = Nothing, returnState = false)
	PROGRESSVERBOSE = parameterDictionary["PROGRESSVERBOSE"]
	PROGRESSVERBOSE::Bool

	NUMGROUPS = parameterDictionary["NUMGROUPS"]
	NUMGROUPS::Int
	NUMAGENTSPERNORM = parameterDictionary["NUMAGENTSPERNORM"]
	NUMAGENTSPERNORM::Int
	NUMAGENTS = length(NORMS) * NUMAGENTSPERNORM
	NUMGENERATIONS = parameterDictionary["NUMGENERATIONS"]
	NUMGENERATIONS::Int
	BATCHSPERAGENT = parameterDictionary["BATCHSPERAGENT"]
	BATCHSPERAGENT::Int
	BATCHSIZE = parameterDictionary["BATCHSIZE"]
	BATCHSIZE::Int
	# INTERACTIONSPERAGENT = 100
	INTERACTIONSPERAGENT = BATCHSIZE * BATCHSPERAGENT
	NUMIMITATE = parameterDictionary["NUMIMITATE"]
	NUMIMITATE::Int

	# Eobs = 0.02
	Ecoop = parameterDictionary["Ecoop"]
	Ecoop::Float64
	w = parameterDictionary["w"]
	w::Float64
	ustrat = parameterDictionary["ustrat"]
	ustrat::Float64
	utype = parameterDictionary["utype"]
	utype::Float64
	# u01 = 0.0 #type mutation rate 0 -> 1
	# u10 = 0.0 #type mutation rate 1 -> 0
	gameBenefit = parameterDictionary["gameBenefit"]
	gameBenefit::Float64

	gameCost = parameterDictionary["gameCost"]
	gameCost::Float64
	intergroupUpdateP = parameterDictionary["intergroupUpdateP"]
	intergroupUpdateP::Float64
	perpetratorNorms = parameterDictionary["perpetratorNorms"]
	perpetratorNorms::Bool
	relativeNorms = parameterDictionary["relativeNorms"]
	relativeNorms::Bool
	uvisibility = parameterDictionary["uvisibility"]
	uvisibility::Float64
	imitationCoupling = parameterDictionary["imitationCoupling"]
	imitationCoupling::Float64
	typeImitate = parameterDictionary["typeImitate"]
	typeImitate::Bool
	establishEquilibrium = parameterDictionary["establishEquilibrium"]
	establishEquilibrium::Bool
	updateMethod = parameterDictionary["updateMethod"]
	updateMethod::String 


	if establishEquilibrium
		finalu = uvisibility
		uvisibility = 0.0
	end


	if state == Nothing
		# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
		population = [Agent(i%NUMGROUPS, i + 1, (i % length(NORMS)) + 1) for i in 0:(NUMAGENTS-1)]
		# arguments are: type, ID, normNumber (referring to index in norm list)

		# println("empathy matrix: $empathyMatrix")

		# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
		reputations = LinearAlgebra.ones(Int8, length(NORMS), NUMAGENTS)
		# reputations = SharedArray{Int8,2}((length(NORMS), NUMAGENTS))

		tmpArr = [0 for i in 1:NUMGROUPS]

		statistics = [ (LinearAlgebra.zeros(Float64, length(NORMS), NUMGROUPS), 0.0, tmpArr) for i in 1:NUMGENERATIONS]
		statistics::Array{Tuple{Array{Float64,2}, Float64, Array{Int,1}},1}

		state = EvolutionState(population, reputations, statistics)
	else
		population = state.population
		reputations = state.reputations
		statistics = state.statistics
	end

	groupSets = calculateGroupSets(population, NUMGROUPS)

	mostRecentImgMatrix = LinearAlgebra.zeros(Float64, length(NORMS), length(NORMS))

	oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	for n in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)

		cooperationRate = 0.0
		cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT


		if establishEquilibrium && (n > 3000)
			uvisibility = finalu
		end


		for i in 1:NUMAGENTS * BATCHSPERAGENT

			cooperationRate = batchUpdate!(NUMAGENTS, BATCHSIZE, reputations, population, perpetratorNorms, relativeNorms, uvisibility, oneShotMatrix, roundPayoffs, cooperationRate)
			
		end

		cooperationRate = cooperationRate / cooperationRateDenominator

		generateStatistics!(statistics, n, population, NUMGROUPS, cooperationRate)


		#imitation update. intergroupUpdateP calculations rely on parity of indices originating in original
		#creation of population. Beware changing that.

		if n == NUMGENERATIONS
			mostRecentImgMatrix = imageMatrix(reputations, population)
		end
		if updateMethod == "imitationUpdate"
			imitationUpdate!(population, roundPayoffs, groupSets, NUMIMITATE, intergroupUpdateP, typeImitate, w, imitationCoupling)
		elseif updateMethod == "deathBirthUpdate"
			deathBirthUpdate(population, roundPayoffs, NUMIMITATE, w)
		else
			println("BAD UPDATE METHOD: $updateMethod")
		end

		mutatePopulation!(population, ustrat)
		
		endTime = time_ns()
		elapsedSecs = (endTime - startTime) / 1.0e9
		if PROGRESSVERBOSE
			println("**Completed modeling generation: $n in $elapsedSecs seconds")
			# println("statistics for this generration are: $(statistics[i])")
		end

	end
	println("Computing image matrix")
	mostRecentImgMatrix = imageMatrix(reputations, population)

	println("Completed Simulation")
	if returnState
		rstate = EvolutionState(population, reputations, statistics)
		return rstate, mostRecentImgMatrix
	else
		return statistics, mostRecentImgMatrix
	end
end

function testEvolve()
	

	if TEST
		stats = evolve(testPopParams, testEnvParams, testNorm, LinearAlgebra.zeros(Float64, 2,2))
		println(stats[end])
	else 
		return evolve(testPopParams, testEnvParams, testNorm, LinearAlgebra.zeros(Float64, 2,2))
	end

end
# defaultPopParams = Dict("numAgents" => 100, "numGenerations" => 150000)
# defaultEnvParams = Dict("Ecoop" => 0.0, "Eobs" => 0.0, "ustrat" => 0.001, "u01" => 0.0, "u10" => 0.0, "w" => 1.0,
# 					"gameBenefit" => 5.0, "gameCost" => 1.0, )

# defaultNorm = LinearAlgebra.ones(Int,2,2)
# defaultNorm[1,2] = 0


if TEST
	testEvolve()
end

