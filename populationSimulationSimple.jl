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


function listDoubler(l)

	for i in 1:4
		append!(l,l)
	end
	return l
end

# const NORMS = [genNorm(i) for i in 0:(16 - 1)]
# const NORMS = listDoubler([genNorm(i) for i in 0:(16 - 1)])
# for i in 1:4
# 	append!(NORMS, NORMS)
# end
const NORMMAX = 16

println("Number of norms: $NORMMAX")


function getRepMask(action,rep,typeaction,typerep)
	return (1 << (2*action + rep + 4*(2*typeaction + typerep))&63) 
end

function getRep(normNumber, action, rep, typeaction, typerep)
	return (normNumber & _getRepMask(action,rep,typeaction,typerep)) != 0
end

# function makeAgent(type, ID, normNumber)
# 	# numString = base(4,normNumber, 4)
# 	# norms = [NORMS[parse(Int,s)] for s in numString]
# 	myAgent = Agent(type, ID, normNumber)
# 	return myAgent
# end

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
	typeProportions = LinearAlgebra.zeros(Float64, NORMMAX)
	

	for j in 1:length(population)
		typeProportions[population[j].normNumber] += 1

	end
	typeProportions = typeProportions / length(population)


	statList[generation] = (typeProportions, cooperationRate)
	return statList

end

function imageMatrix(reputations, population)

	numNorms = length(NORMS)
	numAgents = size(reputations)[2] #assumption about indexing here

	imageMatrix = LinearAlgebra.zeros(Float64, numNorms, numNorms)

	for i in 1:numNorms
		reps = LinearAlgebra.zeros(Float64, numNorms, 2)
		for n in 1:length(population)

			reps[population[n].normNumber,1] += reputations[i,n]
			reps[population[n].normNumber,2] += 1.0

		end
		reps[:,1] = reps[:,1] ./ reps[:,2]
		imageMatrix[i,:] = reps[:,1]
	end
	return imageMatrix
end


function updateReps!(reputations, a,b,action)
	for j in 1:length(NORMS)
		jsview = reputations[j,b]
		# normInd = 2 * population[agentID].type + population[adversaryID].type
		newrep = NORMS[j][1][action + 1, jsview + 1]
		reputations[j,a] = newrep
	end
end



function evolve()
	PROGRESSVERBOSE = true

	NUMAGENTSPERNORM = 200
	NUMAGENTS = NORMMAX * NUMAGENTSPERNORM
	NUMGENERATIONS = 1500
	BATCHSPERAGENT = 2
	BATCHSIZE = 50
	# INTERACTIONSPERAGENT = 100
	INTERACTIONSPERAGENT = BATCHSIZE * BATCHSPERAGENT
	NUMIMITATE = 40

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
	population = [Agent(0, i + 1, (i % NORMMAX) + 1) for i in 0:(NUMAGENTS-1)]
	# arguments are: type, ID, normNumber (referring to index in norm list)

	# println("empathy matrix: $empathyMatrix")

	# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	reputations = LinearAlgebra.ones(Int8, NORMMAX, NUMAGENTS)


	statistics = [ (LinearAlgebra.zeros(Float64, NORMMAX), 0.0) for i in 1:NUMGENERATIONS]
	statistics::Array{Tuple{Array{Float64,1}, Float64},1}

	mostRecentImgMatrix = LinearAlgebra.zeros(Float64, NORMMAX, NORMMAX)


	for n in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)

		cooperationRate = 0.0
		cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

		for i in 1:NUMAGENTS * BATCHSPERAGENT

			a = trunc(Int, ceil(rand() * NUMAGENTS))
			b = 0
			action = 0

			for j in 1:BATCHSIZE

			#agent

				b = trunc(Int, ceil(rand() * NUMAGENTS))
				#adversary

				bRep = reputations[population[a].normNumber, b]

				action = bRep
				# action = moveError(action, Ecoop)

				aPayoff, bPayoff = oneShotMatrix[action + 1]

				roundPayoffs[a] += aPayoff
				roundPayoffs[b] += bPayoff

				if a == b
					baseMask = getRepMask(action, 0, population[a].type, population[b].type)
					masks = baseMask .<< reputations[:,a] 
					reputations[:,a]  = (((1:NORMMAX) .& masks) .!= 0) .& (1 % Int8)

					# for j in 1:NORMMAX
					# 	jsview = reputations[j,b]
					# 	normInd = population[adversaryID].type + 1 #COOL OPTIONS HERE
					# 	newrep = NORMS[j][normInd][action + 1, jsview + 1]
					# 	# newrep = getRep(j, action, jsview, population[a].type, population[b].type)
					# 	reputations[j,a] = newrep
					# end
				end

				cooperationRate += action

			end

			baseMask = getRepMask(action, 0, population[a].type, population[b].type)
			masks = baseMask .<< reputations[:,a]
			reputations[:,a]  = (((1:NORMMAX) .& masks) .!= 0) .& (1 % Int8)

			# updateReps!(reputations, a, b,action)
			# for j in 1:length(NORMS)
			# 	jsview = reputations[j,b]
			# 	normInd = population[adversaryID].type + 1 #COOL OPTIONS HERE
			# 	newrep = NORMS[j][normInd][action + 1, jsview + 1]
			# 	# newrep = getRep(j, action, jsview, population[a].type, population[b].type)
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
				num = rand(1:NORMMAX)
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

