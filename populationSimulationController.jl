import Distributed
using Distributed

addprocs(3)

@everywhere import LinearAlgebra
@everywhere include("populationSimulation.jl")

function envArrayToDict(array)
	keyList = ["Ecoop", "Eobs", "ustrat", "u01", "u10", "w", "gameBenefit", "gameCost"]
	return Dict(keyList[i] => array[i] for i in 1:length(keyList))
end

function popArrayToDict(array)
	keyList = ["numAgents", "numGenerations"]
	return Dict(keyList[i] => array[i] for i in 1:length(keyList))
end



function evolveDistributed(parameterTuples)
	parameterTuples = [(popArrayToDict(x[1]), envArrayToDict(x[2]), x[3], x[4]) for x in parameterTuples]
	println("Julia got parameter tuples: $parameterTuples" )
	return Distributed.pmap(x->evolve(x...), parameterTuples)
end



println(Distributed.nprocs())

# defaultPopParams = Dict("numAgents" => 100, "numGenerations" => 150000)
# defaultEnvParams = Dict("Ecoop" => 0.0, "Eobs" => 0.0, "ustrat" => 0.001, "u01" => 0.0, "u10" => 0.0, "w" => 1.0,
# 					"gameBenefit" => 5.0, "gameCost" => 1.0, )

# defaultNorm = LinearAlgebra.ones(Int,2,2)
# defaultNorm[1,2] = 0

