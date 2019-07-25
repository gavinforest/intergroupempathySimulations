import LinearAlgebra


DEBUG = true
PROGRESSVERBOSE = true
TEST = false



# const SS = [[1, 0], [1, 1]]
# const SJ = [[1, 0], [0, 1]]
# const SC = [[0, 0], [1, 1]]
# const SH = [[0, 0], [0, 1]]

# const NORMS = [SC, SH, SJ, SS]
# const STRATNUMS = collect(1:4)
# const STRATNAMES = ["SC", "SH", "SJ", "SS"]



mutable struct Agent 
	type::Int
	ID::Int
	normNumber::Int
end

function genSingleNorm(num)
	nums = digits(num, base = 2, pad = 4)
	mat = LinearAlgebra.zeros(Int, 2,2)
	mat[1,1] = nums[1]
	mat[2,2] = nums[4]
	mat[2,1] = nums[3]
	mat[1,2] = nums[2]

	# mat = [[nums[4], nums[3]],[nums[2], nums[1]]]
	return mat
end

function genNorm(num)
	ordered = reverse(digits(num, base=16))
	return tuple([genSingleNorm(i) for i in ordered]...)
end

function genTerminationMatrix(mat)
	termMatrix = LinearAlgebra.zeros(Int8, 2)
	for i in 1:2
		if mat[i][1] == mat[i][2]
			termMatrix[i] = 1

	return termMatrix
end


const NORMS = [genNorm(i) for i in 0:(16 - 1)]
const TERMINATIONMATRICES = [genTerminationMatrix(mat) for mat in NORMS]



function makeAgent(type, ID, normNumber)
	# numString = base(4,normNumber, 4)
	# norms = [NORMS[parse(Int,s)] for s in numString]
	myAgent = Agent(type, ID, normNumber)
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
	typeProportions = LinearAlgebra.zeros(Float64, length(NORMS))
	

	for j in 1:length(population)
		typeProportions[population[j].normNumber + 1] += 1

	end
	typeProportions = typeProportions / length(population)


	statList[generation] = (typeProportions, cooperationRate)
	return statList

end

function imageMatrix(reputations)

	numNorms = length(NORMS)
	numAgents = size(reputations)[2] #assumption about indexing here

	imageMatrix = LinearAlgebra.zeros(Float64, numNorms, numNorms)

	for i in 1:numNorms
		for j in 1:numNorms
			avg = 0.0
			avgDenom = 0.0

			for n in range(j, step = numNorms, stop = numAgents) # once again relying on modding during population creation
				avg += reputations[i,n]
				avgDenom += 1.0
			end

			imageMatrix[i,j] = avg/avgDenom

		end
	end

mutable struct interaction
	actor::Int
	recipient::Int
	action::Int
	stepN::Int
	recipientHead
	parents::Int
	children::Int
end


function cleanInteractions(numAgents)
	return [interaction(i,i,-1, 0, Nothing, 0,0) for i in 1:numAgents]
end

function calculateReputation(i, normNumber, biglist,checkpointedReps)
	termMatrix = TERMINATIONMATRICES[normNumber]
	headObj = biglist[i]
	children = headObj.children
	interactionSequence = [headObj]

	obj = headObj
	while obj.children > 0 && termMatrix[obj.action + 1] == 0
		obj = obj.recipientHead
		append!(interactionSequence, obj)
	end

	#the important aspect of termination matrices here is that the reputatioin no longer matters,
	#so we can assume that the final object refers to the checkpointed reputation, because if it doesn't,
	#then we must have terminated early, and therefore the reputation doesn't matter for the norm calculation
	#anyway
	

	recipientRep = checkpointedReps[normNumber,interactionSequence[end].recipient] 

	for i in length(interactionSequence):-1:1
		action = interactionSequence[i].action
		recipientRep = NORMS[i][actioni,recipientRep]

	return recipientRep


end

function refreshReputations(allInteractions, previousCheckpoints)
	for i in 1:length(allInteractions)
		


end






function evolve()
	PROGRESSVERBOSE = true

	NUMAGENTSPERNORM = 100
	NUMAGENTS = length(NORMS) * NUMAGENTSPERNORM
	NUMGENERATIONS = 1
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
	population = [makeAgent(i % 2, i + 1, (i % length(NORMS)) + 1) for i in 0:(NUMAGENTS-1)]
	# arguments are: type, ID, normNumber (referring to index in norm list)

	# println("empathy matrix: $empathyMatrix")

	# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	reputations = LinearAlgebra.ones(Int8, length(NORMS), NUMAGENTS)

	Interactions = cleanInteractions(NUMAGENTS)
	Interactions::Array{interaction, 1}

	statistics = [ (LinearAlgebra.zeros(Float64, 256), 0.0) for i in 1:NUMGENERATIONS]
	statistics::Array{Tuple{Array{Float64,1}, Float64},1}


	for n in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)

		cooperationRate = 0.0
		cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

		for i in 1:NUMAGENTS * INTERACTIONSPERAGENT

			a = ceil(rand() * NUMAGENTS)
			#agent

			b = ceil(rand() * NUMAGENTS)
			#adversary

			bRep = reputations[population[a].normNumber, b]

			action = bRep
			action = moveError(action, Ecoop)

			aPayoff, bPayoff = oneShotMatrix[action + 1]

			roundPayoffs[a] += aPayoff
			roundPayoffs[b] += bPayoff
			
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

		cooperationRate = cooperationRate / cooperationRateDenominator

		statistics = generateStatistics!(statistics, generation, population, cooperationRate)

		for j in 1:NUMAGENTS
			for a in 1:INTERACTIONSPERAGENT
				reputations[j, real(reputationUpdates[j,a])] = imag(reputationUpdates[j,a])


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

