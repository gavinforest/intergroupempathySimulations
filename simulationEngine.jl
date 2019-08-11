using Distributed
import LinearAlgebra
import StatsBase

# ------ Local File Imports ------

using simulationStructs
using simulationUpdateRules
using simulationAgentInteractions
using cacheTools




const NORMS = [genNorm(i) for i in 0:(16^2 -1)]
const DEBUG = true
const PROGRESSVERBOSE = true


function mirrorNormNumber(num)
	return ((num >> 4) + (num<<4))&255
end



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



println("Number of norms: $(length(NORMS))")
println("Number of processes: $(nprocs())")

function evolve(simulationParameters, runParameters, processSpecs, cacheChannel, state = Nothing, returnState = false,)
	PROGRESSVERBOSE = simulationParameters["PROGRESSVERBOSE"]
	PROGRESSVERBOSE::Bool

	NUMGROUPS = simulationParameters["NUMGROUPS"]
	NUMGROUPS::Int
	NUMAGENTSPERNORM = simulationParameters["NUMAGENTSPERNORM"]
	NUMAGENTSPERNORM::Int
	NUMAGENTS = length(NORMS) * NUMAGENTSPERNORM
	NUMGENERATIONS = simulationParameters["NUMGENERATIONS"]
	NUMGENERATIONS::Int
	BATCHSPERAGENT = simulationParameters["BATCHSPERAGENT"]
	BATCHSPERAGENT::Int
	BATCHSIZE = simulationParameters["BATCHSIZE"]
	BATCHSIZE::Int
	# INTERACTIONSPERAGENT = 100
	INTERACTIONSPERAGENT = BATCHSIZE * BATCHSPERAGENT
	NUMIMITATE = simulationParameters["NUMIMITATE"]
	NUMIMITATE::Int

	# Eobs = 0.02
	Ecoop = simulationParameters["Ecoop"]
	Ecoop::Float64
	w = simulationParameters["w"]
	w::Float64
	ustrat = simulationParameters["ustrat"]
	ustrat::Float64
	utype = simulationParameters["utype"]
	utype::Float64
	# u01 = 0.0 #type mutation rate 0 -> 1
	# u10 = 0.0 #type mutation rate 1 -> 0
	gameBenefit = simulationParameters["gameBenefit"]
	gameBenefit::Float64

	gameCost = simulationParameters["gameCost"]
	gameCost::Float64
	intergroupUpdateP = simulationParameters["intergroupUpdateP"]
	intergroupUpdateP::Float64
	perpetratorNorms = simulationParameters["perpetratorNorms"]
	perpetratorNorms::Bool
	relativeNorms = simulationParameters["relativeNorms"]
	relativeNorms::Bool
	uvisibility = simulationParameters["uvisibility"]
	uvisibility::Float64
	imitationCoupling = simulationParameters["imitationCoupling"]
	imitationCoupling::Float64
	typeImitate = simulationParameters["typeImitate"]
	typeImitate::Bool
	establishEquilibrium = simulationParameters["establishEquilibrium"]
	establishEquilibrium::Bool
	updateMethod = simulationParameters["updateMethod"]
	updateMethod::String 

	cachePeriod = NUMGENERATIONS
	if "cachePeriod" in processSpecs
		cachePeriod = processSpecs["cachePeriod"]
	end


	if establishEquilibrium
		finalu = uvisibility
		uvisibility = 0.0
	end

	statistics = [ (LinearAlgebra.zeros(Float64, length(NORMS), NUMGROUPS), 0.0, tmpArr) for i in 1:NUMGENERATIONS]
	statistics::Array{Tuple{Array{Float64,2}, Float64, Array{Int,1}},1}

	if "startState" in runParameters
		startName = runParameters["startState"]["name"]
		startID = runParameters["startState"]["ID"]
		startGen = runParameters["startState"]["generation"]
		state = getState(startName, startID, startGen)

	elseif state == Nothing
		# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
		population = [Agent(i%NUMGROUPS, i + 1, (i % length(NORMS)) + 1) for i in 0:(NUMAGENTS-1)]
		# arguments are: type, ID, normNumber (referring to index in norm list)

		# println("empathy matrix: $empathyMatrix")

		# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
		reputations = LinearAlgebra.ones(Int8, length(NORMS), NUMAGENTS)
		# reputations = SharedArray{Int8,2}((length(NORMS), NUMAGENTS))

		tmpArr = [0 for i in 1:NUMGROUPS]


		state = EvolutionState(population, reputations, statistics)
	end

	state.statistics = statistics
	population = state.population
	reputations = state.reputations

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

			cooperationRate = batchUpdate!(NUMAGENTS, BATCHSIZE, reputations, population, perpetratorNorms, relativeNorms, uvisibility, oneShotMatrix, roundPayoffs, cooperationRate, NORMS)
			
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

		mutatePopulation!(population, ustrat, NORMS)
		
		endTime = time_ns()
		elapsedSecs = (endTime - startTime) / 1.0e9
		if PROGRESSVERBOSE
			println("**Completed modeling generation: $n in $elapsedSecs seconds")
			# println("statistics for this generration are: $(statistics[i])")
		end

		if n % cachePeriod == 0
			cacheState = EvolutionState(population, reputations, statistics[(n - cachePeriod):n])
			toCache = (processSpecs["name"], processSpecs["processID"], n, cacheState)
			put!(cacheChannel, toCache)
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


