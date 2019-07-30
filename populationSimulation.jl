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
	mat = LinearAlgebra.zeros(Int8, 2,2)
	mat[1,1] = nums[1] % Int8
	mat[2,2] = nums[4] % Int8
	mat[2,1] = nums[3] % Int8
	mat[1,2] = nums[2] % Int8

	# mat = [[nums[4], nums[3]],[nums[2], nums[1]]]
	return mat
end

function genNorm(num)
	ordered = reverse(digits(num, base=16))
	return tuple([genSingleNorm(i) for i in ordered]...)
end

function genTerminationMatrix(mat)
	termMatrix = LinearAlgebra.ones(Int8, 2) * (-1)
	for i in 1:2
		if mat[i,1] == mat[i,2]
			termMatrix[i] = mat[i,1]
		end
	end

	return termMatrix
end


NORMS = [genNorm(i) for i in 0:(16 - 1)]
for i in 1:4
	append!(NORMS, NORMS)
end
const TERMINATIONMATRICES = [genTerminationMatrix(mattuple[1]) for mattuple in NORMS]
const TERMINATIONLIST = [Set{Int}([j for j in 1:length(TERMINATIONMATRICES) if TERMINATIONMATRICES[j][i] != -1]) for i in 1:2]



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


function _getRepMask(action,rep,typeaction,typerep)
	return (1 << (2*action + rep + 4*(2*typeaction + typerep)))
end

function getRep(normNumber, action, rep, typeaction, typerep)
	return (normNumber & _getRepMask(action,rep,typeaction,typerep)) != 0
end


# function typeStratDoubleToNum(firstTup, secondTup)
# 	(firstType, firstStratNumber) = firstTup
# 	(secondType, secondStratNumber) = secondTup
# 	return 6 * typeStratToNum((firstType, firstStratNumber)) + typeStratToNum((secondType, secondStratNumber))
# end

function generateStatistics!(statList,generation, population, cooperationRate)
	typeProportions = LinearAlgebra.zeros(Float64, length(NORMS))
	

	for j in 1:length(population)
		typeProportions[population[j].normNumber ] += 1

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
end


mutable struct interaction
	actor::Int
	actorNorm::Int
	recipient::Int
	action::Int
	# stepN::Int
	recipientHead
	children::Int
end


function cleanInteractions(numAgents)
	return [interaction(i,-1,i,-1, Nothing,0) for i in 1:numAgents]
end


function iterateBack(normNumber, interactionSequence, recipientRep)
	offset = 0
	if interactionSequence[end] == -1
		offset = 1
	end
	norm = NORMS[normNumber][1]
	for i in length(interactionSequence) - offset:-1:1
		# action = interactionSequence[i].action
		action = interactionSequence[i]
		# if action == -1
		# 	println("Somehow still ended up with action -1 with offset $offset")
		# end
		recipientRep = norm[action+1,recipientRep + 1]
	end
	return norm[interactionSequence[1]+1,recipientRep + 1]
end

function calculateReputation(i, normNumber, biglist,checkpointedReps)
	if biglist[i].children == 0
		return checkpointedReps[normNumber,i] 
	end
	headObj = biglist[i]

	#can now assume nontrivial interaction sequence
	interactionSequence = [headObj.action]
	termMatrix = TERMINATIONMATRICES[normNumber]

	obj = headObj
	while obj.children > 0 && termMatrix[obj.action + 1] == -1 && obj.actorNorm != normNumber
		obj = obj.recipientHead
		push!(interactionSequence, obj.action)
	end

	#the important aspect of termination matrices here is that the reputatioin no longer matters,
	#so we can assume that the final object refers to the checkpointed reputation, because if it doesn't,
	#then we must have terminated early, and therefore the reputation doesn't matter for the norm calculation
	#anyway
	if obj.actorNorm != normNumber
		recipientRep = checkpointedReps[normNumber,obj.recipient] 
	else
		recipientRep = obj.action
	end

	return iterateBack(normNumber, interactionSequence, recipientRep)

end




function refreshReputations(allInteractions, previousCheckpoints)
	newReps = LinearAlgebra.zeros(Int, size(previousCheckpoints)...)
	for i in 1:length(allInteractions)
		headObj = allInteractions[i]
		interactionSequence = [headObj.action]

		terminated = Set()
		reps = LinearAlgebra.zeros(Int, length(NORMS))

		norms = Set{Int64}(1:length(NORMS))
		obj = headObj
		while obj.children > 0 && length(norms) > 0
			push!(interactionSequence, obj.action)
			# push!(interactionSequence, obj)

			if !  in(obj.action, terminated) && obj.action != -1
				terminating = TERMINATIONLIST[obj.action + 1]
					
				for num in setdiff(terminating, terminated)
					recipientRep = 1 #doesn't matter if terminating

					reps[num] = iterateBack(num, interactionSequence, recipientRep)
				end

				norms = setdiff!(norms, terminating)
				push!(terminated, obj.action)
			end

			if in(obj.actorNorm, norms) 
				recpientRep = obj.action
				reps[obj.actorNorm] = iterateBack(obj.actorNorm, interactionSequence, recpientRep)
				delete!(norms, obj.actorNorm)
			end

			obj = obj.recipientHead
		end


		for norm in norms
			recipientRep = previousCheckpoints[norm,obj.recipient] 
			
			reps[norm] = iterateBack(norm, interactionSequence, recipientRep)

		end

		newReps[:,i] = reps

	end
	return cleanInteractions(size(allInteractions)[1]), newReps

end






function evolve()
	PROGRESSVERBOSE = true

	NUMAGENTSPERNORM = 200
	NUMAGENTS = length(NORMS) * NUMAGENTSPERNORM
	NUMGENERATIONS = 1500
	BATCHSPERAGENT = 10
	BATCHSIZE = 10
	# INTERACTIONSPERAGENT = 100
	INTERACTIONSPERAGENT = BATCHSIZE * BATCHSPERAGENT
	INTERACTIONSPERGEN = INTERACTIONSPERAGENT * NUMAGENTSPERNORM * length(NORMS)
	NUMIMITATE = 40
	REFRESHTIME = 3

	# Eobs = 0.02
	Ecoop = 0.00
	w = 1.0
	ustrat = 0.0005
	# u01 = 0.0 #type mutation rate 0 -> 1
	# u10 = 0.0 #type mutation rate 1 -> 0
	gameBenefit = 8
	gameCost = 1.0
	intergroupUpdateP = 0.0


	oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
	population = [Agent(i % 2, i + 1, (i % length(NORMS)) + 1) for i in 0:(NUMAGENTS-1)]
	# arguments are: type, ID, normNumber (referring to index in norm list)

	# println("empathy matrix: $empathyMatrix")

	# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	checkpointReputations = LinearAlgebra.ones(Int8, length(NORMS), NUMAGENTS)

	Interactions = cleanInteractions(NUMAGENTS)
	Interactions::Array{interaction, 1}

	statistics = [ (LinearAlgebra.zeros(Float64, 256), 0.0) for i in 1:NUMGENERATIONS]
	statistics::Array{Tuple{Array{Float64,1}, Float64},1}


	for n in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)

		cooperationRate = 0.0
		cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT


		if n % REFRESHTIME == 0
			Interactions, checkpointReputations = refreshReputations(Interactions, checkpointReputations)
		end

		for i in 1:NUMAGENTS * BATCHSPERAGENT

			# if i % (NUMAGENTS / 10) * BATCHSPERAGENT == 0
			# 	Interactions, checkpointReputations = refreshReputations(Interactions, checkpointReputations)
			# end

			a = trunc(Int, ceil(rand() * NUMAGENTS))
			anorm = population[a].normNumber
			b = 0
			action = 0

			for j in 1:BATCHSIZE

			#agent

				b = trunc(Int, ceil(rand() * NUMAGENTS))
				#adversary

				bRep = calculateReputation(b, anorm, Interactions, checkpointReputations)

				action = bRep
				# action = moveError(action, Ecoop)

				aPayoff, bPayoff = oneShotMatrix[action + 1]

				roundPayoffs[a] += aPayoff
				roundPayoffs[b] += bPayoff

				if a == b
					Interactions[a] = interaction(a,anorm, b ,action, Interactions[a], Interactions[a].children + 1)
				end

				cooperationRate += action

			end

			Interactions[a] = interaction(a,anorm,b,action, Interactions[a], Interactions[b].children + 1)
			# updateReps!(reputations, a, b,action)
			# for j in 1:length(NORMS)
			# 	jsview = reputations[j,b]
			# 	# normInd = 2 * population[agentID].type + population[adversaryID].type
			# 	newrep = NORMS[j][1][action + 1, jsview + 1]
			# 	reputations[j,a] = newrep
			# end


			# newrep = population[j].norms[normInd][agentAction, judgesview]

			#strategy specific coop rate calculation
		end

		cooperationRate = cooperationRate / cooperationRateDenominator

		statistics = generateStatistics!(statistics, n, population, cooperationRate)


		#imitation update. intergroupUpdateP calculations rely on parity of indices originating in original
		#creation of population. Beware changing that.

		if n == NUMGENERATIONS
			mostRecentImgMatrix = imageMatrix(reputations, population)
		end

		changed = Set()

		for i in 1:NUMIMITATE
			ind1 = trunc(Int,floor(rand() * NUMAGENTS))
			ind2 = trunc(Int, floor(rand() * NUMAGENTS))

			if ind1 != ind2 && ! in(ind1 + 1, changed) && ! in(ind2 + 1, changed)


				# inter = rand() < intergroupUpdateP

				# if (ind2 - ind1) %2 == 0 && inter
				# 	ind2 = (ind2 + 1) % NUMAGENTS
				# elseif (ind2 - ind1) %2 == 1 && !inter
				# 	ind2 = (ind2 + 1) % NUMAGENTS
				# 	if ind1 == ind2
				# 		ind2 = (ind2 + 2) % NUMAGENTS
				# 	end
				# end
					
				ind1 += 1
				ind2 += 1

				pCopy = 1.0 / (1.0 + exp( (- w) * (roundPayoffs[ind2] - roundPayoffs[ind1])))
				if rand() < pCopy
					population[ind1].normNumber = population[ind2].normNumber
					push!(changed, ind1)
				end
			end
		end



		if PROGRESSVERBOSE
			println("--- simulated generation")
		end

		#random drift applied uniformly to all individuals
		for j in 1:NUMAGENTS
			if rand() < ustrat
				num = rand(1:length(NORMS))
				population[j].normNumber = num
			end

			# if population[j].type == 0 && u01 > rand()
			# 	population[j].type = 1
			# elseif population[j].type == 1 && u10 > rand()
			# 	population[j].type = 0
			# end
		end
		endTime = time_ns()
		elapsedSecs = (endTime - startTime) / 1.0e9
		if PROGRESSVERBOSE
			println("**Completed modeling generation: $n in $elapsedSecs seconds")
			# println("statistics for this generration are: $(statistics[i])")
		end

	end


	println("Completed Simulation")
	return statistics, mostRecentImgMatrix
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

