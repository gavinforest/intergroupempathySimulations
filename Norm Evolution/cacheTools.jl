module cacheTools

export cacheState, getState, setupCache


include("simulationStructs.jl")
using .simulationStructs
using HDF5
import JSON

const cacheDir = "./cached/"
const ledgerFilename = cacheDir * "ledger.json"

function genFilename(root)
	ts = string(time_ns())
	return cacheDir * ts * root * ".jld"
end


function agentToDict(agent)
	mydict = Dict("type" => agent.type, "normNumber" => agent.normNumber, "ID" => agent.ID)
	return mydict
end

function dictToAgent(agentDict)
	return Agent(agentDict["type"], agentDict["ID"], agentDict["normNumber"])
end

function populationToDictionary(population)
	# population::Array{Agent,1}
	println("making population dictionary")
	return [agentToDict(agent) for agent in population]
end


function statisticsToArrayDict(statistics)
	# println("pre type assert")
	statistics::Array{Tuple{Array{Float64,2}, Float64, Array{Int,1}},1}
	# println("post type assert")
	# println("length statistics: $(length(statistics))")
	proportions = [tup[1] for tup in statistics]
	cooperationRates = [tup[2] for tup in statistics]
	typeNums = [tup[3] for tup in statistics]
	d = Dict("proportions" => proportions, "cooperationRates" => cooperationRates, "typeNums" => typeNums)
	# println("made statistics array dict")
	return d
end

function arrayDictToStats(arrayDict)
	stats = []
	for i in 1:length(arrayDict["cooperationRates"])
		tup = (arrayDict["proportions"][i], arrayDict["cooperationRates"][i], arrayDict["typeNums"][i])
		append!(stats, tup)
	end

	return stats
end



function saveState(filename, evoState)
	h5open(filename * ".reputations.hdf5", "w") do file
		write(file, "reputations", evoState.reputations)
		println("Caching -- saved reputation")
	end
	open(filename * ".json", "w") do file
		println("caching -- starting stat dict")
		statDict = statisticsToArrayDict(evoState.statistics)
		println("caching -- made stat dict")
		popDict = populationToDictionary(evoState.population)
		println("caching -- made pop dict")
		d = Dict("statistics" => statDict, "population" => popDict)
		println("Caching -- generating dictionary JSON")
		JSON.print(file, d)
	end
end

function readState(filename)
	reputations = h5open(filename * ".reputations.hdf5", "r") do file
		read(file, "reputations")
	end
	d = JSON.parsefile(filename * ".json")
	stats = d["statistics"]
	population = d["population"]
	population = map(dictToAgent, population)
	statistics = arrayDictToStates(d["statistics"])
	return EvolutionState(population, reputations, statistics)
end

function readStackStates(filenames)
	states = map(readState, filenames)
	pop = stats[end].population
	reps = stats[end].reputations
	stats = cat(1, map(x -> x.statistics, states)...)
	return EvolutionState(pop, reps, stats)
end

function searchDir(dir, name)
	everything = readdir(dir)
	if length(everything) == 0
		return false
	end
	occurences = map(x -> x == name, everything)
	if length(occurences) == 1
		return occurences[1]
	else
		return reduce((y,z) -> y || z, occurences)
	end
end

function setupCache(name, ID, simparameters, logisticparams)
	IDstring = string(ID)
	if ! searchDir(cacheDir, name)
		mkdir(cacheDir * name)
	end
	if ! searchDir(cacheDir * name, IDstring)
		mkdir(cacheDir * name * "/" * IDstring)
	end

	ledger = JSON.parsefile(ledgerFilename)
	if ! haskey(ledger, name)
		ledger[name] = Dict()
	end
	if ! haskey(ledger[name], IDstring)
		ledger[name][IDstring] = Dict()
		ledger[name][IDstring]["simparameters"] = simparameters
		ledger[name][IDstring]["runparameters"] = logisticparams
		ledger[name][IDstring]["generations"] = Dict()
		open(ledgerFilename, "w") do file
			JSON.print(file, ledger)
			println("JSON printed to ledger")
		end
	else 
		return "Already created"
	end
	return "Success"
end



function addToLedger(name, ID, generation, filename)
	IDstring = string(ID)
	println("starting to add to ledger")
	ledger = JSON.parsefile(ledgerFilename)
	ledger[name][IDstring]["generations"][generation] = filename
	open(ledgerFilename, "w") do file
		JSON.print(file, ledger)
	end
end

function cacheState(name, ID, generation, state)
	IDstring = string(ID)
	filename = cacheDir * name * "/" * IDstring * "/" * string(time_ns())
	saveState(filename, state)
	prinln("caching -- saved state")
	addToLedger(name, ID, generation, filename)
	println("caching -- added to ledger")
end

function getFilename(name, ID, generation)
	IDstring = string(ID)
	ledger = JSON.parsefile(ledgerFilename)
	return ledger[name][IDsring]["generations"][generation]

end

function getState(name, ID, generation)
	fname = getFilename(name, ID, generation)
	return readState(fname)
end


end


