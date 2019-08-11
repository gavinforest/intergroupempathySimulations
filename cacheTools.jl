using simulationStructs
using HDF5
import JSON

const cacheDir = "./cached/"
const ledgerFilename = cacheDir + "ledger.json"

function genFilename(root)
	ts = string(time_ns())
	return cacheDir + ts + root + ".jld"
end


function agentToDict(agent)
	mydict = Dict("type" => agent.type, "normNumber" => agent.normNumber, "ID" => agent.ID)
	return mydict
end

function dictToAgent(agentDict)
	return Agent(agentDict["type"], agentDict["ID"], agentDict["normNumber"])
end

function populationToDictionary(population)
	population::Array{Agent,1}
	return [agentToDict(agent) for agent in population]
end


function statisticsToArrayDict(statistics)
	statistics::Array{Tuple{Array{Float64,2}, Float64, Array{Int,1}},1}
	proportions = [tup[1] for tup in statistics]
	cooperationRates = [tup[2] for tup in statistics]
	typeNums = [tup[3] for tup in statistics]
	d = Dict("proportions" => proportions, "cooperationRates" => cooperationRates, "typeNums" => typeNums)
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
	h5open(filename + ".reputations.hdf5", "w") do file
		write(file, "reputations", evoState.reputations)
	end
	open(filename + ".json", "w") do file
		d = Dict("statistics" => statisticsToArrayDict(statistics), "population" => populationToDictionary(population))
		JSON.print(file, d)
	end
end

function readState(filename)
	reputations = h5open(filename + ".reputations.hdf5", "r") do file
		read(file, "reputations")
	end
	d = JSON.parsefile(filename + ".json")
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


function setupCache(name, ID, simparameters, logisticparams)
	if ! name in readdir(cacheDir)
		mkdir(cacheDir + name)
	end
	if ! string(ID) in readdir(cacheDir + name)
		mkdir(cacheDir + name + "/" + string(ID))
		ledger = JSON.parsefile(ledgerFilename)
		if ! name in ledger
			ledger[name] = Dict()
		end
		if ! ID in ledger[name]
			ledger[name][ID] = Dict()
			ledger["simparameters"] = simparameters
			ledger["runparameters"] = logisticparams
			ledger["generations"] = Dict()
			open(ledgerFilename, "w") do file
				JSON.print(file, ledger)
			end
		else 
			return "Already created"
		end
	end
	return "Success"
end



function addToLedger(name, ID, generation, filename)
	ledger = JSON.parsefile(ledgerFilename)
	ledger[name][ID]["generations"][generation] = filename
	open(ledgerFilename, "w") do file
		JSON.print(file, ledger)
	end
end

function cacheState(name, ID, generation, state)
	filename = cacheDir + name + "/" + string(ID) + "/" + string(time_ns())
	saveState(filename, state)
	addToLedger(name, ID, generation, filename)
end

function getFilename(name, ID, generation)
	ledger = JSON.parsefile(ledgerFilename)
	return ledger[name][ID]["generations"][generation]

end

function getState(name, ID, generation)
	fname = getFilename(name, ID, generation)
	return readState(fname)
end



