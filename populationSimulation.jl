import LinearAlgebra


DEBUG = true
PROGRESSVERBOSE = true
TEST = false



const SS = [[1, 0], [1, 1]]
const SJ = [[1, 0], [0, 1]]
const SC = [[0, 0], [1, 1]]
const SH = [[0, 0], [0, 1]]

const NORMS = [SC, SH, SJ, SS]
const STRATNUMS = collect(1:4)
const STRATNAMES = ["SC", "SH", "SJ", "SS"]

mutable struct Agent 
	type::Int
	ID::Int
	norms::Tuple{Vararg{Array{Int,2},4}}
	normNumber::Int
end

function makeAgent(type, ID, normNumber)
	numString = base(4,normNumber, 4)
	norms = [NORMS[parse(Int,s)] for s in numString]
	myAgent = Agent(type, ID, norms, normNumber)
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



# function typeStratDoubleToNum(firstTup, secondTup)
# 	(firstType, firstStratNumber) = firstTup
# 	(secondType, secondStratNumber) = secondTup
# 	return 6 * typeStratToNum((firstType, firstStratNumber)) + typeStratToNum((secondType, secondStratNumber))
# end

function generateStatistics!(statList,generation, population, cooperationRate)
	typeProportions = LinearAlgebra.zeros(Float64, 256)
	

	for j in 1:length(population)
		typeProportions[population[j].normNumber + 1] += 1

	end
	typeProportions = typeProportions / length(population)


	statList[generation] = (typeProportions, cooperationRate)
	return statList

end


function evolve()
	PROGRESSVERBOSE = true

	NUMAGENTSPERNORM = 100
	NUMAGENTS = length(NORMS) ^ 4 * NUMAGENTSPERNORM
	NUMGENERATIONS = 100000
	INTERACTIONSPERAGENT = 100

	Eobs = 0.02
	Ecoop = 0.02
	w = 1.0
	ustrat = 0.0025
	u01 = 0.0 #type mutation rate 0 -> 1
	u10 = 0.0 #type mutation rate 1 -> 0
	gameBenefit = 5.0
	gameCost = 1.0
	intergroupUpdateP = 0.0


	oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
	population = [makeAgent(i % 2, i, i % length(NORMS)) for i in 1:NUMAGENTS]
	# arguments are: type, ID, strategy number, empathy

	# println("empathy matrix: $empathyMatrix")

	reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	# reputations = LinearAlgebra.zeros(Int, NUMAGENTS, NUMAGENTS)

	statistics = [ (LinearAlgebra.zeros(Float64, 256), 0.0) for i in 1:NUMGENERATIONS]
	statistics::Array{Tuple{Array{Float64,1}, Float64},1}


	for i in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)
		reputationUpdates = LinearAlgebra.zeros(Complex{Int}, NUMAGENTS, INTERACTIONSPERAGENT)

		cooperationRate = 0.0
		cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

		for j in 1:NUMAGENTS
			#judges

			for a in 1:INTERACTIONSPERAGENT

				agentID = ceil(rand() * NUMAGENTS)
				#agents

				adversaryID = ceil(rand() * NUMAGENTS)
				#randomly selected but agents never have to play themselves.

				adversaryRep = reputations[agentID, adversaryID]

				agentAction = adversaryRep
				agentAction = moveError(agentAction, Ecoop)

				agentPayoff, adversaryPayoff = oneShotMatrix[agentAction + 1]

				roundPayoffs[agentID] += agentPayoff
				roundPayoffs[adversaryID] += adversaryPayoff

				judgetype = population[j].type
				
				judgesview = reputations[j,adversaryID]

				normInd = 2 * population[agentID].type + population[adversaryID].type

				newrep = population[j].norms[normInd][agentAction, judgesview]

				if rand()<Eobs
					if newrep ==1
						newrep = 0
					else
						newrep = 1
					end
				end


				reputationUpdates[j,a] = agentID + newrep * im


				#strategy specific coop rate calculation
				cooperationRate += agentAction


			end
		end

		cooperationRate = cooperationRate / cooperationRateDenominator

		statistics = generateStatistics!(statistics, generation, population, cooperationRate)

		for j in 1:NUMAGENTS
			for a in 1:INTERACTIONSPERAGENT
				reputations[j, Real(reputationUpdates[j,a])] = Im(reputationUpdates[j,a])

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

