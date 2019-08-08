using Distributed
import LinearAlgebra

addprocs(5)

@everywhere begin
    include("populationSimulationSimple.jl")

	function singleRun(parameters)
		starting = time_ns()
		println("Starting Job")
		res = evolve(parameters)
		t = (time_ns() - starting) / 1.0e9
		println("Processed Job in $t seconds")
		return res
	end

	function evolveFromState(parameters, state)
		starting = time_ns()
		println("Starting Job")
		res = evolve(parameters, state= state)
		t = (time_ns() - starting) / 1.0e9
		println("Processed Job in $t seconds")
		return res
	end

	function generateStateFromFrequencies(normFrequencies, groupNumbers, parameters)
		population = []
		numNorms = sizeof(normFrequencies)[1]
		numGroups = length(groupNumbers)
		numAgents = sum(groupNumbers)

		distributions = [[round(normFrequencies[i,j] * groupNumbers[j]) for i in 1:sizeof(normFrequencies)[1]] for j in 1:length(groupNumbers)]

		ID = 1
		for i in 0:(length(groupNumbers) -1)
			for j in 1:length(distributions[i + 1])
				for k in 1:distributions[i,j]
					append!(population, Agent(i, ID, j - 1))
					ID += 1
				end
			end
		end
		reputations = LinearAlgebra.ones(Int8, numNorms, numAgents)
		tmpArr = [0 for i in 1:numGroups]
		statistics = [ (LinearAlgebra.zeros(Float64, numNorms, numGroups), 0.0, tmpArr) for i in 1:5]

		return EvolutionState(population, reputations, statistics)
	end

	function evolveFromFrequencies(parameters, normFrequencies, groupNumbers)

		starting = time_ns()
		println("Starting Job")
		println("Generating state from frequencies")
		state = generateStateFromFrequencies(normFrequencies, groupNumbers, parameters)
		t = (time_ns() - starting) / 1.0e9
		println("Generated state from frequencies in $t seconds")
		
		#zero things out so only updating reputation
		tmpParams = copy(parameters)
		tmpParams["NUMIMITATE"] = 0
		tmpParams["ustrat"] = 0.0
		tmpParams["utype"] = 0.0
		tmpParams["NUMGENERATIONS"] = 5

		println("Time evolving reputations with fixed population")

		state, m = evolve(tmpParams, state=state, returnState = true)
		t = (time_ns() - t) / 1.0e9
		println("Evolved reputations in $t seconds")
		println("Starting evolution")

		res = evolve(parameters, state = state)

		t = (time_ns() - starting) / 1.0e9
		println("Processed Job in $t seconds")

		return res
	end







end



function evolveDistributed(dictionaryList)
	return Distributed.pmap(singleRun, dictionaryList)
end
		
	

	
