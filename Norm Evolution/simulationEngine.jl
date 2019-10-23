module simulationEngine

export evolve

using Distributed
import LinearAlgebra
import StatsBase

# ------ Local File Imports ------

include("simulationStructs.jl")
include("simulationUpdateRules.jl")
include("simulationAgentInteractions.jl")
include("cacheTools.jl")
include("parameterTools.jl")
include("simulationUtilities.jl")

using .simulationUpdateRules
using .simulationAgentInteractions
using .cacheTools
using .parameterTools
using .simulationUtilities
using .simulationStructs




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
	#Constants
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
	INTERACTIONSPERAGENT = BATCHSIZE * BATCHSPERAGENT
	NUMIMITATE = simulationParameters["NUMIMITATE"]
	NUMIMITATE::Int

	cachePeriod = NUMGENERATIONS
	if haskey(runParameters, "cachePeriod")
		cachePeriod = runParameters["cachePeriod"]
		cachePeriod::Int
	end
	println("Cacheperiod: $cachePeriod")


	tmpArr = [0 for i in 1:NUMGROUPS]
	statistics = [ (LinearAlgebra.zeros(Float64, length(NORMS), NUMGROUPS), 0.0, tmpArr) for i in 1:NUMGENERATIONS]
	statistics::Array{Tuple{Array{Float64,2}, Float64, Array{Int,1}},1}

	if  haskey(runParameters, "startState")
		startName = runParameters["startState"]["name"]
		startID = runParameters["startState"]["ID"]
		startGen = runParameters["startState"]["generation"]
		state = getState(startName, startID, startGen)

	elseif state == Nothing
		# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.


		population = [Agent(i%NUMGROUPS, i + 1, (i % length(NORMS)) + 1) for i in 0:(NUMAGENTS-1)]
		population::Array{Agent, 1}
		# arguments are: type, ID, normNumber (referring to index in norm list)

		# println("empathy matrix: $empathyMatrix")

		# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
		reputations = LinearAlgebra.ones(Int8, length(NORMS), NUMAGENTS)
		# reputations = SharedArray{Int8,2}((length(NORMS), NUMAGENTS))



		state = EvolutionState(population, reputations, statistics)
	end

	state.statistics = statistics
	population = state.population
	reputations = state.reputations

	groupBounds = sortPopulation!(state.population, state.reputations, NUMGROUPS)
	println("groupBounds: $groupBounds")
	# groupSets = calculateGroupSets(population, NUMGROUPS)

	mostRecentImgMatrix = LinearAlgebra.zeros(Float64, length(NORMS), length(NORMS))


	for n in 1:NUMGENERATIONS

		if haskey(runParameters, string(n))
			simulationParameters = modifyParameters(simulationParameters, state, runParameters[n])
		end

		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)

		cooperationRate = 0.0
		cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

		egalitarianAgentInteractions = true
		for v in simulationParameters["groupWeights"]
			for j in simulationParameters["groupWeights"]
				if v != j
					egalitarianAgentInteractions = false
				end
			end
		end

		for i in 1:NUMAGENTS * BATCHSPERAGENT

			cooperationRate = batchUpdate!(simulationParameters, reputations, population, groupBounds, roundPayoffs, cooperationRate, NORMS, egalitarianAgentInteractions)
			
		end

		cooperationRate = cooperationRate / cooperationRateDenominator

		generateStatistics!(statistics, n, population, NUMGROUPS, cooperationRate)


		#imitation update. intergroupUpdateP calculations rely on parity of indices originating in original
		#creation of population. Beware changing that.

		if n == NUMGENERATIONS
			mostRecentImgMatrix = imageMatrix(reputations, population)
		end
		if simulationParameters["updateMethod"] == "pairwiseComparison"
			pairwiseComparison!(population, roundPayoffs, groupBounds, NUMIMITATE, simulationParameters)
		elseif simulationParameters["updateMethod"] == "imitationUpdate"
			imitationUpdate!(population, roundPayoffs, NUMIMITATE, simulationParameters)
		elseif simulationParameters["updateMethod"] == "deathBirthUpdate"
			deathBirthUpdate!(population, roundPayoffs, NUMIMITATE, simulationParameters)
		elseif simulationParameters["updateMethod"] == "birthDeathUpdate"
			birthDeathUpdate!(populationl, roundPayoffs, NUMIMITATE, simulationParameters)
		else
			println("BAD UPDATE METHOD: $updateMethod")
		end

		mutatePopulation!(population, simulationParameters["ustrat"], NORMS)

		groupBounds = sortPopulation!(population,reputations, NUMGROUPS)
		
		endTime = time_ns()
		elapsedSecs = (endTime - startTime) / 1.0e9
		if PROGRESSVERBOSE
			println("**Completed modeling generation: $n in $elapsedSecs seconds")
			# println("statistics for this generration are: $(statistics[i])")
		end

		if (n % cachePeriod == 1) && (n > 1)
			println("Caching")
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


end

