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

end



function evolveDistributed(dictionaryList)
	return Distributed.pmap(singleRun, dictionaryList)
end
		
	

	
