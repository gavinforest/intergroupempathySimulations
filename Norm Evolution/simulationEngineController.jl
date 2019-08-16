push!(LOAD_PATH, "/Users/gavin/Documents/Stay Loose/Research/Evolutionary Dynamics/Inter-group empathy/Integroup Empathy Repository/Norm Evolution")

using Distributed
import LinearAlgebra

include("cacheTools.jl")
import .cacheTools
# include("simulationEngine.jl")
# using .simulationEngine

addprocs(5)

@everywhere begin
	push!(LOAD_PATH, "/Users/gavin/Documents/Stay Loose/Research/Evolutionary Dynamics/Inter-group empathy/Integroup Empathy Repository/Norm Evolution/")

    include("simulationEngine.jl")
    import .simulationEngine

	function singleRun(parameters)
		starting = time_ns()
		println("Starting Job")
		res = simulationEngine.evolve(parameters...)
		t = (time_ns() - starting) / 1.0e9
		println("Processed Job in $t seconds")
		return res
	end

	# function evolveFromState(parameters, state)
	# 	starting = time_ns()
	# 	println("Starting Job")
	# 	res = evolve(parameters, state= state)
	# 	t = (time_ns() - starting) / 1.0e9
	# 	println("Processed Job in $t seconds")
	# 	return res
	# end

	# function generateStateFromFrequencies(normFrequencies, groupNumbers, parameters)
	# 	population = []
	# 	numNorms = sizeof(normFrequencies)[1]
	# 	numGroups = length(groupNumbers)
	# 	numAgents = sum(groupNumbers)

	# 	distributions = [[round(normFrequencies[i,j] * groupNumbers[j]) for i in 1:sizeof(normFrequencies)[1]] for j in 1:length(groupNumbers)]

	# 	ID = 1
	# 	for i in 0:(length(groupNumbers) -1)
	# 		for j in 1:length(distributions[i + 1])
	# 			for k in 1:distributions[i,j]
	# 				append!(population, Agent(i, ID, j - 1))
	# 				ID += 1
	# 			end
	# 		end
	# 	end
	# 	reputations = LinearAlgebra.ones(Int8, numNorms, numAgents)
	# 	tmpArr = [0 for i in 1:numGroups]
	# 	statistics = [ (LinearAlgebra.zeros(Float64, numNorms, numGroups), 0.0, tmpArr) for i in 1:5]

	# 	return EvolutionState(population, reputations, statistics)
	# end

	# function evolveFromFrequencies(parameters, normFrequencies, groupNumbers)

	# 	starting = time_ns()
	# 	println("Starting Job")
	# 	println("Generating state from frequencies")
	# 	state = generateStateFromFrequencies(normFrequencies, groupNumbers, parameters)
	# 	t = (time_ns() - starting) / 1.0e9
	# 	println("Generated state from frequencies in $t seconds")
		
	# 	#zero things out so only updating reputation
	# 	tmpParams = copy(parameters)
	# 	tmpParams["NUMIMITATE"] = 0
	# 	tmpParams["ustrat"] = 0.0
	# 	tmpParams["utype"] = 0.0
	# 	tmpParams["NUMGENERATIONS"] = 5

	# 	println("Time evolving reputations with fixed population")

	# 	state, m = evolve(tmpParams, state=state, returnState = true)
	# 	t = (time_ns() - t) / 1.0e9
	# 	println("Evolved reputations in $t seconds")
	# 	println("Starting evolution")

	# 	res = evolve(parameters, state = state)

	# 	t = (time_ns() - starting) / 1.0e9
	# 	println("Processed Job in $t seconds")

	# 	return res
	# end

end

function cachingLoop(cacheChannel)
	while true
		toCache = take!(cacheChannel)
		cacheTools.cacheState(toCache...)
		println("Cached in caching loop")
	end
end

function makeRunArgument(name,i, dict, channel)
	return (dict["simparameters"], dict["runparameters"], Dict("name" => name, "processID" => i), channel)
end

function evolveDistributed(name, dictionaryList)
	cacheChannel = RemoteChannel(() -> Channel{Tuple}(10))

	# mapable = (i,x) -> makeRunArgument(name, i, x, cacheChannel)

	args = [makeRunArgument(name, i , x, cacheChannel) for (i,x) in  enumerate(dictionaryList)]

	println("Beginning to set up cache for run")
	for arg in args
		ret = cacheTools.setupCache(name, arg[3]["processID"], arg[1], arg[2])
		if ret == "Already created"
			println("Dangerous situation -- setupCache returned already created on:")
			println("name: $name")
			println("ID: $(arg[3]["processID"])")
			println("Quitting!")
			return
		end
	end
		
	println("Set up caches")

	@async cachingLoop(cacheChannel)

	println("Spun off caching loop with async")

	return Distributed.pmap(singleRun, args)
end
		
	

	
