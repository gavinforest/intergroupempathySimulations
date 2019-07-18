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

function typeStratToNum(tup)
	(type, stratNumber) = tup
	return 3 * type + stratNumber - 1 #-1 from stratNumber being in [1,2,3]
end

# function typeStratDoubleToNum(firstTup, secondTup)
# 	(firstType, firstStratNumber) = firstTup
# 	(secondType, secondStratNumber) = secondTup
# 	return 6 * typeStratToNum((firstType, firstStratNumber)) + typeStratToNum((secondType, secondStratNumber))
# end

function generateStatistics!(statList, population, reputations, generation, cooperationRateArray, cooperationRateDetailedArray, roundPayoffs)
	typeProportions = LinearAlgebra.zeros(Float64, 2, 4)
	repStats = LinearAlgebra.zeros(Float64, 6,6)
	repStatsDenoms = LinearAlgebra.zeros(Float64, 6,6)
	payoffsByTypeStrat = LinearAlgebra.zeros(Float64, 6, 1)

	for j in 1:length(population)
		jnum = typeStratToNum(population[j].type, population[j].stratNumber) + 1


		typeProportions[(population[j].type + 1), population[j].stratNumber] += 1.0 
		typeProportions[(population[j].type + 1), 4] += 1.0 

		payoffsByTypeStrat[jnum] += roundPayoffs[j]

		for k in 1:length(population)
			knum = typeStratToNum(population[k].type, population[k].stratNumber) + 1

			repStats[jnum, knum] += reputations[j,k] * 1.0 
			repStatsDenoms[jnum, knum] += 1.0 

		end

	end

	repStats = repStats ./ repStatsDenoms

	for j in 1:3
		payoffsByStrat[j] = payoffsByStrat[j] / (typeProportions[1,j] + typeProportions[2,j])
	end


	statList[generation]["proportions"] = typeProportions
	statList[generation]["reputationsViewFromTo"] = repStats
	statList[generation]["cooperationRate"] = cooperationRateArray
	statList[generation]["cooperationRateDetailed"] = cooperationRateDetailedArray[:,:,1]
	statList[generation]["roundPayoffs"] = payoffsByTypeStrat

	return statList

end


function evolve(populationParameters::Dict{String, Int}, environmentParameters::Dict{String, Float64}, norm0::Array{Int,2}, norm1::Array{Int,2},empathyMatrix::Array{Float64,2}, printing::Bool)::Array{Dict{String, Array{Float64, 2}}, 1}
	PROGRESSVERBOSE = printing

	NUMAGENTS = populationParameters["numAgents"]
	NUMAGENTS::Int
	NUMGENERATIONS = populationParameters["numGenerations"]
	NUMGENERATIONS::Int
	NORM0 = norm0 #norm that type 0 individuals use
	NORM1 = norm1 #norm that type 1 individuals use
	NORMS = [NORM0, NORM1]

	Eobs = environmentParameters["Eobs"]
	Eobs::Float64 #Observeration Error Rate. Effects judging
	Ecoop = environmentParameters["Ecoop"]
	Ecoop::Float64 #Cooperation Error Rate. Effects actions
	w = environmentParameters["w"]
	w::Float64 #selection strength
	ustrat = environmentParameters["ustrat"]
	ustrat::Float64 #random drift in strategy
	u01 = environmentParameters["u01"]
	u01::Float64 #type mutation rate 0 -> 1
	u10 = environmentParameters["u10"]
	u10::Float64 #type mutation rate 1 -> 0
	gameBenefit = environmentParameters["gameBenefit"]
	gameBenefit::Float64
	gameCost = environmentParameters["gameCost"]
	gameCost:: Float64
	intergroupUpdateP = environmentParameters["intergroupUpdateP"]
	intergroupUpdateP::Float64 #probability that the target individual to imitate is from the other group


	oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
	population = [makeAgent(i % 2, i, rand([1,2,3]), empathyMatrix) for i in 1:NUMAGENTS]
	# arguments are: type, ID, strategy number, empathy

	# println("empathy matrix: $empathyMatrix")

	reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	# reputations = LinearAlgebra.zeros(Int, NUMAGENTS, NUMAGENTS)

	statistics = [ Dict{String, Array{Float64, 2}}() for i in 1:NUMGENERATIONS]


	for i in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)
		reputationUpdates = LinearAlgebra.zeros(Int, NUMAGENTS, NUMAGENTS)

		cooperationRate = LinearAlgebra.zeros(Float64, 9, 2)
		cooperationRateDetailed = LinearAlgebra.zeros(Float64, 6,6,2)

		for j in 1:NUMAGENTS
			#judges

			for a in 1:NUMAGENTS
				#agents

				adversaryID = rand((a+1):(a + NUMAGENTS - 1)) % NUMAGENTS + 1
				#randomly selected but agents never have to play themselves.

				adversaryRep = reputations[a, adversaryID]

				agentAction = population[a].strategy[adversaryRep + 1]
				agentAction = moveError(agentAction, Ecoop)

				agentPayoff, adversaryPayoff = oneShotMatrix[agentAction + 1]

				roundPayoffs[a] += agentPayoff
				roundPayoffs[adversaryID] += adversaryPayoff

				judgetype = population[j].type
				
				oldrep = reputations[j,a]
				newrep = 0 #just to initialize variable

				if rand() < population[j].empathy[population[a].type + 1]
					newrep = NORMS[judgetype+1][agentAction + 1, adversaryRep + 1]
				else
					newrep = NORMS[judgetype+1][agentAction+1, reputations[j,adversaryID] + 1]
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

				#type specific coop rate calculation
				binaryIndex = 2 * population[a].type + population[adversaryID].type

				coopertionRate[5 + binaryIndex,1] += agentAction
				cooperationRate[5 + binaryIndex, 2] += 1.0

				#total intergroup coop rate calculation (difficult to calculate from above given stochastic pairing)
				if binaryIndex == 1 || binaryIndex == 2
					cooperationRate[9,1] += agentAction
					cooperationRate[9,2] += 1.0
				end

				#detailed (type,strat) -> (type,strat) cooperation rate 
				anum = typeStratToNum((population[a].type, population[a].stratNumber))
				advnum = typeStratToNum((population[adversaryID].type, population[adversaryID].stratNumber))
				cooperationRateDetailed[anum, advnum, 1] += agentAction
				cooperationRateDetailed[anum,advnum, 2] += 1.0 


			end
		end

		cooperationRate[:,1] = cooperationRate[:,1] ./ cooperationRate[:,2]
		cooperationRateDetailed[:,:,1] = cooperationRate[:,:,1] ./ cooperationRate[:,:,2]

		statistics = generateStatistics!(statistics, population, reputations, i, cooperationRate,cooperationRateDetailed, roundPayoffs)

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

		#random drift applied uniformly to all individuals
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

