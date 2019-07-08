import LinearAlgebra
const LA = LinearAlgebra

DEBUG = true
PROGRESSVERBOSE = true

const ALLC = [1,1]
const DISC = [0,1]
const ALLD = [0,0]

const STRATEGIES = [ALLD, DISC, ALLC]
const STRATNUMS = collect(1:3)
const STRATNAMES = ["ALLD", "DISC", "ALLD"]

mutable struct Agent 
	type::Int
	ID::Int
	empathy::AbstractArray{Float64, 1}
	strategy::AbstractArray{Int, 1}
	stratString::String 
	stratNumber::Int8
end

function stratToString(strat::AbstractArray{Int, 1})
	stratInd = sum(strat) + 1
	return STRATNAMES[stratInd]
end

function stratToNumber(strat::AbstractArray{Int,1})
	return sum(strat) + 1
end

function makeAgent(type, ID, stratNumber)
	strat = STRATEGIES[stratNumber]
	myAgent = Agent(type, ID, [0.0,0.0], strat, stratToString(strat), stratNumber)
	return myAgent
end

function moveError(move, ec)
	if move == 0
		return 0
	elseif ec < rand()
		return 0
	else 
		return 1
	end
end

function generateStatistics!(statList, population, reputations, generation, cooperationRateArray)
	typeProportions = LA.zeros(Float64, 2, 4)
	repStats = LA.zeros(Float64, 3,3)
	repStatsDenoms = LA.zeros(Float64, 3,3)

	for j in 1:length(population)
		typeProportions[(population[j].type + 1), population[j].stratNumber] = 1.0 + typeProportions[(population[j].type + 1), population[j].stratNumber]
		typeProportions[(population[j].type + 1), 4] = 1.0 + typeProportions[(population[j].type + 1), 4]

		for k in 1:length(population)
			repStats[population[j].stratNumber, population[k].stratNumber] = reputations[j,k] * 1.0 + repStats[population[j].stratNumber, population[k].stratNumber]
			repStatsDenoms[population[j].stratNumber, population[k].stratNumber] = 1.0 + repStatsDenoms[population[j].stratNumber, population[k].stratNumber]
		end

	end

	repStats = repStats ./ repStatsDenoms

	statList[generation]["proportions"] = typeProportions
	statList[generation]["reputationsViewFromTo"] = repStats
	statList[generation]["cooperationRate"] = cooperationRateArray

	return statList

end



function evolve(populationParameters::Dict{String, Int}, environmentParameters::Dict{String, Float64}, norm::Array{Int,2})::Array{Dict{String, Array{Float64, 2}}, 1}
	NUMAGENTS = populationParameters["numAgents"]
	NUMAGENTS::Int
	NUMGENERATIONS = populationParameters["numGenerations"]
	NUMGENERATIONS::Int
	NORM = norm

	Eobs = environmentParameters["Eobs"]
	Eobs::Float64
	Ecoop = environmentParameters["Ecoop"]
	Ecoop::Float64
	w = environmentParameters["w"]
	w::Float64
	ustrat = environmentParameters["ustrat"]
	ustrat::Float64
	u01 = environmentParameters["u01"]
	u01::Float64
	u10 = environmentParameters["u10"]
	u10::Float64
	gameBenefit = environmentParameters["gameBenefit"]
	gameBenefit::Float64
	gameCost = environmentParameters["gameCost"]
	gameCost:: Float64

	oneShotMatrix = [(0.0, gameBenefit),(-gameCost, gameBenefit - gameCost)]

	population = [makeAgent(i % 2, i, i % 3 + 1) for i in 1:100]

	reputations = LA.ones(Int, NUMAGENTS, NUMAGENTS) #Might want to make this randomly generated

	statistics = [ Dict{String, Array{Float64, 2}}() for i in 1:NUMGENERATIONS]


	for i in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LA.zeros(Float64, NUMAGENTS)
		reputationUpdates = LA.zeros(Int, NUMAGENTS, NUMAGENTS)

		cooperationRate = LA.zeros(Float64, 4, 2)

		for j in 1:NUMAGENTS

			for a in 1:NUMAGENTS

				adversaryID = rand((a+1):(a + NUMAGENTS - 1)) % NUMAGENTS + 1

				adversaryRep = reputations[a, adversaryID]

				agentAction = population[a].strategy[adversaryRep + 1]
				agentAction = moveError(agentAction, Ecoop)

				agentPayoff, adversaryPayoff = oneShotMatrix[agentAction + 1]

				roundPayoffs[a] = roundPayoffs[a] + agentPayoff
				roundPayoffs[adversaryID] = roundPayoffs[adversaryID] + adversaryPayoff


				oldrep = reputations[j,a]
				local newrep

				if rand() < population[j].empathy[population[a].type + 1]
					newrep = NORM[agentAction + 1, adversaryRep]
				else
					newrep = NORM[agentAction+1, reputations[j,adversaryID]]
				end

				reputationUpdates[j,a] = newrep - oldrep

				#strategy specific coop rate calculation
				cooperationRate[population[a].stratNumber,1] = agentAction + cooperationRate[population[a].stratNumber,1]
				cooperationRate[population[a].stratNumber,2] = 1 + cooperationRate[population[a].stratNumber,2]

				#total coop rate calculation
				cooperationRate[4,1] = agentAction + cooperationRate[4,1]
				cooperationRate[4,2] = 1 + cooperationRate[4,2]

			end
		end

		cooperationRate[:,1] = cooperationRate[:,1] ./ cooperationRate[:,2]

		statistics = generateStatistics!(statistics, population, reputations, i, cooperationRate)

		reputations = reputations + reputationUpdates

		#imitation update 
		ind1 = rand(0:(NUMAGENTS-1))
		ind2 = (ind1 + rand(1:(NUMAGENTS-1))) % NUMAGENTS
		ind1 += 1
		ind2 += 1

		pCopy = 1.0 / (1.0 + exp( -w * (roundPayoffs[ind2] - roundPayoffs[ind1])))
		if pCopy < rand()
			population[ind1].strategy = population[ind2].strategy
		end

		if PROGRESSVERBOSE
			println("--- simulated generation")
		end


		stratNames = ["ALLD","DISC","ALLC"]
		for j in 1:NUMAGENTS
			if rand() < ustrat
				num = rand(STRATNUMS)
				population[j].stratNumber = num
				population[j].strategy = STRATEGIES[num]
				population[j].stratString = STRATNAMES[num]
			end

			if population[j].type == 0 && u01 > rand()
				population[j].type = 1
			elseif population[j].type == 1 && u10 > rand()
				population[j].type = 0
			end
		end
		endTime = time_ns()
		elapsedSecs = (endTime - startTime) / 1.0e9
		if PROGRESSVERBOSE
			println("**Completed modeling generation: $i in $elapsedSecs seconds")
		end

	end

	println("Completed Simulation")
	return statistics
end

testPopParams = Dict("numAgents" => 100, "numGenerations" => 200)

testEnvParams = Dict("Ecoop" => 0.0, "Eobs" => 0.0, "ustrat" => 0.001, "u01" => 0.0, "u10" => 0.0, "w" => 1.0,
				"gameBenefit" => 5.0, "gameCost" => 1.0, )

testNorm = LA.ones(Int,2,2)
testNorm[1,2] = 0

evolve(testPopParams, testEnvParams, testNorm)









