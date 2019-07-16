import LinearAlgebra


DEBUG = true
PROGRESSVERBOSE = true
TEST = false

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

function makeAgent(type, ID, stratNumber, empathyM)
	strat = STRATEGIES[stratNumber]
	myAgent = Agent(type, ID, empathyM[type+1,:], strat, stratToString(strat), stratNumber)
	return myAgent
end

function moveError(move, ec)
	if move == 0
		return 0
	elseif rand() < ec
		return 0
	else 
		return 1
	end
end

function generateStatistics!(statList, population, reputations, generation, cooperationRateArray, roundPayoffs)
	typeProportions = LinearAlgebra.zeros(Float64, 2, 4)
	repStats = LinearAlgebra.zeros(Float64, 3,3)
	repStatsDenoms = LinearAlgebra.zeros(Float64, 3,3)
	payoffsByStrat = LinearAlgebra.zeros(Float64, 3, 1)

	for j in 1:length(population)
		typeProportions[(population[j].type + 1), population[j].stratNumber] += 1.0 
		typeProportions[(population[j].type + 1), 4] += 1.0 

		payoffsByStrat[population[j].stratNumber] += roundPayoffs[j]

		for k in 1:length(population)
			repStats[population[j].stratNumber, population[k].stratNumber] += reputations[j,k] * 1.0 
			repStatsDenoms[population[j].stratNumber, population[k].stratNumber] += 1.0 
		end

	end

	repStats = repStats ./ repStatsDenoms

	for j in 1:3
		payoffsByStrat[j] = payoffsByStrat[j] / (typeProportions[1,j] + typeProportions[2,j])
	end


	statList[generation]["proportions"] = typeProportions
	statList[generation]["reputationsViewFromTo"] = repStats
	statList[generation]["cooperationRate"] = cooperationRateArray
	statList[generation]["roundPayoffs"] = payoffsByStrat

	return statList

end


function evolve(populationParameters::Dict{String, Int}, environmentParameters::Dict{String, Float64}, norm::Array{Int,2},empathyMatrix::Array{Float64,2}, printing::Bool)::Array{Dict{String, Array{Float64, 2}}, 1}
	PROGRESSVERBOSE = printing

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
	intergroupUpdateP = environmentParameters["intergroupUpdateP"]
	intergroupUpdateP::Float64


	oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
	population = [makeAgent(i % 2, i, rand([1,2,3]), empathyMatrix) for i in 1:NUMAGENTS]

	# println("empathy matrix: $empathyMatrix")

	reputations = rand([0,1], NUMAGENTS, NUMAGENTS) #Might want to make this randomly generated
	# reputations = LinearAlgebra.zeros(Int, NUMAGENTS, NUMAGENTS)

	statistics = [ Dict{String, Array{Float64, 2}}() for i in 1:NUMGENERATIONS]


	for i in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)
		reputationUpdates = LinearAlgebra.zeros(Int, NUMAGENTS, NUMAGENTS)

		cooperationRate = LinearAlgebra.zeros(Float64, 6, 2)

		for j in 1:NUMAGENTS

			for a in 1:NUMAGENTS

				adversaryID = rand((a+1):(a + NUMAGENTS - 1)) % NUMAGENTS + 1

				adversaryRep = reputations[a, adversaryID]

				agentAction = population[a].strategy[adversaryRep + 1]
				agentAction = moveError(agentAction, Ecoop)

				agentPayoff, adversaryPayoff = oneShotMatrix[agentAction + 1]

				roundPayoffs[a] += agentPayoff
				roundPayoffs[adversaryID] += adversaryPayoff


				oldrep = reputations[j,a]
				local newrep

				if rand() < population[j].empathy[population[a].type + 1]
					newrep = NORM[agentAction + 1, adversaryRep + 1]
				else
					newrep = NORM[agentAction+1, reputations[j,adversaryID] + 1]
				end

				if rand()<Eobs
					if newrep ==1
						newrep = 0
					else
						newrep = 1
					end
				end

				reputationUpdates[j,a] = newrep - oldrep

				#strategy specific coop rate calculation
				cooperationRate[population[a].stratNumber,1] += agentAction
				cooperationRate[population[a].stratNumber,2] += 1 

				#total coop rate calculation
				cooperationRate[4,1] += agentAction
				cooperationRate[4,2] += 1.0

				if population[a].type != population[adversaryID].type
					cooperationRate[5,1] += agentAction
					cooperationRate[5,2] += 1.0
				else
					cooperationRate[6,1] += agentAction
					cooperationRate[6,2] += 1.0
				end

			end
		end

		cooperationRate[:,1] = cooperationRate[:,1] ./ cooperationRate[:,2]

		statistics = generateStatistics!(statistics, population, reputations, i, cooperationRate, roundPayoffs)

		reputations = reputations + reputationUpdates

		#imitation update. intergroupUpdateP calculations rely on parity of indices originating in original
		#creation of population. Beware changing that.
		ind1 = rand(0:(NUMAGENTS-1))
		ind2 = (ind1 + rand(1:(NUMAGENTS-1))) % NUMAGENTS

		inter = rand() < intergroupUpdateP

		if (ind2 - ind1) %2 == 0 && inter
			ind2 = (ind2 + 1) % NUMAGENTS
		elseif (ind2 - ind1) %2 == 1 && !inter
			ind2 = (ind2 + 1) % NUMAGENTS
			if ind1 == ind2
				ind2 = (ind2 + 2) % NUMAGENTS
			end
		end
			
		ind1 += 1
		ind2 += 1



		pCopy = 1.0 / (1.0 + exp( (- w) * (roundPayoffs[ind2] - roundPayoffs[ind1])))
		if rand() < pCopy
			population[ind1].strategy = population[ind2].strategy
			population[ind1].stratString = population[ind2].stratString
			population[ind1].stratNumber = population[ind2].stratNumber
		end

		if PROGRESSVERBOSE && i%1000==0 && DEBUG
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
		if PROGRESSVERBOSE && i%1000==0
			println("**Completed modeling generation: $i in $elapsedSecs seconds")
			# println("statistics for this generration are: $(statistics[i])")
		end

	end

	println("Completed Simulation")
	return statistics
end

function testEvolve()
	testPopParams = Dict("numAgents" => 100, "numGenerations" => 10000)

	testEnvParams = Dict("Ecoop" => 0.02, "Eobs" => 0.02, "ustrat" => 0.0005, "u01" => 0.0, "u10" => 0.0, "w" => 1.0,
					"gameBenefit" => 5.0, "gameCost" => 1.0)

	testNorm = LinearAlgebra.ones(Int,2,2)
	testNorm[1,2] = 0
	testNorm[2,1] = 1
	println("test norm: $testNorm")

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

