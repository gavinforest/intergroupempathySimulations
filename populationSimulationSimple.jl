using Distributed
import LinearAlgebra


const DEBUG = true
const PROGRESSVERBOSE = true
const TEST = false



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
	ordered = reverse(digits(num, base=16, pad=2))
	return tuple([genSingleNorm(i) for i in ordered]...)
end


# function listDoubler(l)

# 	for i in 1:4
# 		append!(l,l)
# 	end
# 	return l
# end

# const NORMS = [(genNorm(i), genNorm(j)) for i in 0:(16 - 1) for j in 0:(16-1)]
const NORMS = [genNorm(i) for i in 0:(16^2 -1)]
# const NORMS = listDoubler([genNorm(i) for i in 0:(16 - 1)])
# for i in 1:4
# 	append!(NORMS, NORMS)
# end




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

function mirrorNormNumber(num)
	return ((num >> 4) + (num<<4))&255
end



# function typeStratDoubleToNum(firstTup, secondTup)
# 	(firstType, firstStratNumber) = firstTup
# 	(secondType, secondStratNumber) = secondTup
# 	return 6 * typeStratToNum((firstType, firstStratNumber)) + typeStratToNum((secondType, secondStratNumber))
# end

function generateStatistics!(statList,generation, population, numGroups, cooperationRate)
	normProportions = LinearAlgebra.zeros(Float64, length(NORMS), numGroups)
	# type1prop = 0

	for j in 1:length(population)
		normProportions[population[j].normNumber, population[j].type + 1] += 1
		# type1prop += population[j].type 
		

	end

	# type1prop = type1prop / length(population)

	normProportions = normProportions / length(population)

	statList[generation] = (normProportions, cooperationRate)
	# statList[generation] = (normProportions, cooperationRate, 1.0 - type1prop, type1prop)
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


function updateReps!(reputations, population, a,b,action, perpetratorNorms, relativeNorms, uvisibility)
	if relativeNorms
		normInd = 1
		if rand() < uvisibility
			if population[a].type != population[b].type
				normInd = 2
			end
		end
	elseif ! perpetratorNorms
		normInd = population[b].type + 1 #COOL OPTIONS HERE
	else
		normInd = population[a].type + 1
	end

	for j in 1:length(NORMS)
		jsview = reputations[j,b]
		newrep = NORMS[j][normInd][action + 1, jsview + 1]
		reputations[j,a] = newrep
	end
end

function batchUpdate!(numagents,batchsize, reputations, population, perpetratorNorms, relativeNorms, uvisibility, oneShotMatrix, roundPayoffs, cooperationRate)
	a = trunc(Int, ceil(rand() * numagents))
	normNumber = population[a].normNumber
	b = 0
	action = 0

	for j in 1:batchsize
		b = trunc(Int, ceil(rand() * numagents))
		#adversary
		action = reputations[normNumber, b]
		# if !relativeNorms
		# 	action = reputations[normNumber, b]
		# elseif atype
		# 	action = reputations[mirroredNormNumber,b]
		# else
		# 	action = reputations[normNumber, b]
		# action = moveError(action, Ecoop)

		aPayoff, bPayoff = oneShotMatrix[action + 1]

		roundPayoffs[a] += aPayoff
		roundPayoffs[b] += bPayoff

		if a == b
			
			updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility)
		end

		cooperationRate += action

	end

	updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility)

	return cooperationRate
end

# function generation(NUMAGENTS, INTERACTIONSPERAGENT, PROGRESSVERBOSE, reputations, population, NUMIMITATE)
# 	startTime = time_ns()
# 	roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)

# 	cooperationRate = 0.0
# 	cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

# 	for i in 1:NUMAGENTS * BATCHSPERAGENT

# 		cooperationRate = batchUpdate!(NUMAGENTS, BATCHSIZE, reputations, population, perpetratorNorms, oneShotMatrix, roundPayoffs, cooperationRate)
# 	end

# 	cooperationRate = cooperationRate / cooperationRateDenominator

# 	statistics = generateStatistics!(statistics, n, population, cooperationRate)


# 	#imitation update. intergroupUpdateP calculations rely on parity of indices originating in original
# 	#creation of population. Beware changing that.


# 	changed = Set()

# 	for i in 1:NUMIMITATE
# 		ind1 = trunc(Int,floor(rand() * NUMAGENTS))
# 		ind2 = trunc(Int, floor(rand() * NUMAGENTS))

# 		if ind1 != ind2 && ! in(ind1 + 1, changed) && ! in(ind2 + 1, changed)


# 			ind1 += 1
# 			ind2 += 1

# 			pCopy = 1.0 / (1.0 + exp( (- w) * (roundPayoffs[ind2] - roundPayoffs[ind1])))
# 			if rand() < pCopy
# 				population[ind1].normNumber = population[ind2].normNumber
# 				push!(changed, ind1)
# 			end
# 		end
# 	end

# 	#random drift applied uniformly to all individuals
# 	for j in 1:NUMAGENTS
# 		if rand() < ustrat
# 			num = rand(1:length(NORMS))
# 			population[j].normNumber = num
# 		end

# 		# if population[j].type == 0 && u01 > rand()
# 		# 	population[j].type = 1
# 		# elseif population[j].type == 1 && u10 > rand()
# 		# 	population[j].type = 0
# 		# end
# 	end
# 	endTime = time_ns()
# 	elapsedSecs = (endTime - startTime) / 1.0e9
# 	if PROGRESSVERBOSE
# 		println("**Completed modeling generation: $n in $elapsedSecs seconds")
# 		# println("statistics for this generration are: $(statistics[i])")
# 	end


# end


println("Number of norms: $(length(NORMS))")
println("Number of processes: $(nprocs())")

function evolve(parameterDictionary)
	PROGRESSVERBOSE = parameterDictionary["PROGRESSVERBOSE"]
	PROGRESSVERBOSE::Bool

	NUMGROUPS = parameterDictionary["NUMGROUPS"]
	NUMGROUPS::Int
	NUMAGENTSPERNORM = parameterDictionary["NUMAGENTSPERNORM"]
	NUMAGENTSPERNORM::Int
	NUMAGENTS = length(NORMS) * NUMAGENTSPERNORM
	NUMGENERATIONS = parameterDictionary["NUMGENERATIONS"]
	NUMGENERATIONS::Int
	BATCHSPERAGENT = parameterDictionary["BATCHSPERAGENT"]
	BATCHSPERAGENT::Int
	BATCHSIZE = parameterDictionary["BATCHSIZE"]
	BATCHSIZE::Int
	# INTERACTIONSPERAGENT = 100
	INTERACTIONSPERAGENT = BATCHSIZE * BATCHSPERAGENT
	NUMIMITATE = parameterDictionary["NUMIMITATE"]
	NUMIMITATE::Int

	# Eobs = 0.02
	Ecoop = parameterDictionary["Ecoop"]
	Ecoop::Float64
	w = parameterDictionary["w"]
	w::Float64
	ustrat = parameterDictionary["ustrat"]
	ustrat::Float64
	utype = parameterDictionary["utype"]
	utype::Float64
	# u01 = 0.0 #type mutation rate 0 -> 1
	# u10 = 0.0 #type mutation rate 1 -> 0
	gameBenefit = parameterDictionary["gameBenefit"]
	gameBenefit::Float64

	gameCost = parameterDictionary["gameCost"]
	gameCost::Float64
	intergroupUpdateP = parameterDictionary["intergroupUpdateP"]
	intergroupUpdateP::Float64
	perpetratorNorms = parameterDictionary["perpetratorNorms"]
	perpetratorNorms::Bool
	relativeNorms = parameterDictionary["relativeNorms"]
	relativeNorms::Bool
	uvisibility = parameterDictionary["uvisibility"]
	uvisibility::Float64
	imitationCoupling = parameterDictionary["imitationCoupling"]
	imitationCoupling::Float64



	oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
	population = [Agent(i%NUMGROUPS, i + 1, (i % length(NORMS)) + 1) for i in 0:(NUMAGENTS-1)]
	# arguments are: type, ID, normNumber (referring to index in norm list)

	# println("empathy matrix: $empathyMatrix")

	# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	reputations = LinearAlgebra.ones(Int8, length(NORMS), NUMAGENTS)
	# reputations = SharedArray{Int8,2}((length(NORMS), NUMAGENTS))


	statistics = [ (LinearAlgebra.zeros(Float64, length(NORMS), NUMGROUPS), 0.0) for i in 1:NUMGENERATIONS]
	statistics::Array{Tuple{Array{Float64,2}, Float64},1}

	mostRecentImgMatrix = LinearAlgebra.zeros(Float64, length(NORMS), length(NORMS))


	for n in 1:NUMGENERATIONS
		startTime = time_ns()
		roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)

		cooperationRate = 0.0
		cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

		for i in 1:NUMAGENTS * BATCHSPERAGENT

			cooperationRate = batchUpdate!(NUMAGENTS, BATCHSIZE, reputations, population, perpetratorNorms, relativeNorms, uvisibility, oneShotMatrix, roundPayoffs, cooperationRate)

			# a = trunc(Int, ceil(rand() * NUMAGENTS))
			# b = 0
			# action = 0

			# for j in 1:BATCHSIZE

			# #agent

			# 	b = trunc(Int, ceil(rand() * NUMAGENTS))
			# 	#adversary

			# 	bRep = reputations[population[a].normNumber, b]

			# 	action = bRep
			# 	# action = moveError(action, Ecoop)

			# 	aPayoff, bPayoff = oneShotMatrix[action + 1]

			# 	roundPayoffs[a] += aPayoff
			# 	roundPayoffs[b] += bPayoff

			# 	if a == b
			# 		# for j in 1:length(NORMS)
			# 		# 	jsview = reputations[j,b]
			# 		# 	if ! perpetratorNorms
			# 		# 		normInd = population[b].type + 1 #COOL OPTIONS HERE
			# 		# 	else
			# 		# 		normInd = population[a].type + 1
			# 		# 	end

			# 		# 	newrep = NORMS[j][normInd][action + 1, jsview + 1]
			# 		# 	reputations[j,a] = newrep
			# 		# end
			# 		updateReps!(reputations, a, b, action, perpetratorNorms)
			# 	end

			# 	cooperationRate += action

			# end

			# # updateReps!(reputations, a, b,action)
			# # updates = SharedArray{Int8,1}((length(NORMS)))
			# # for j in 1:length(NORMS)
			# # 	jsview = reputations[j,b]
			# # 	if ! perpetratorNorms
			# # 		normInd = population[b].type + 1 #COOL OPTIONS HERE
			# # 	else
			# # 		normInd = population[a].type + 1
			# # 	end
			# # 	# println("NORMS[j]: $(NORMS[j])")
			# # 	newrep = NORMS[j][normInd][action + 1, jsview + 1]
			# # 	reputations[j,a] = newrep
			# # end
			# updateReps!(reputations, a, b, action, perpetratorNorms)
			# reputations[:,a] = updates


			# newrep = population[j].norms[normInd][agentAction, judgesview]

			#strategy specific coop rate calculation
		end

		cooperationRate = cooperationRate / cooperationRateDenominator

		statistics = generateStatistics!(statistics, n, population, NUMGROUPS, cooperationRate)


		#imitation update. intergroupUpdateP calculations rely on parity of indices originating in original
		#creation of population. Beware changing that.

		if n == NUMGENERATIONS
			mostRecentImgMatrix = imageMatrix(reputations, population)
		end

		changed = Set()

		for i in 1:NUMIMITATE
			ind1 = trunc(Int,floor(rand() * NUMAGENTS))
			ind2 = trunc(Int, floor(rand() * NUMAGENTS))

			inter = rand() < intergroupUpdateP

			#CHECK THIS, NOT SURE IF WORKS
			if (ind2 - ind1) % NUMGROUPS == 0 && inter
				ind2 = (ind2 + 1) % NUMAGENTS
			elseif (ind2 - ind1) % NUMGROUPS != 0 && !inter
				ind2 = (ind2 + (NUMGROUPS - (ind2 - ind1) % NUMGROUPS)) % NUMAGENTS
				if ind1 == ind2
					ind2 = (ind2 + NUMGROUPS) % NUMAGENTS
				end
			end

			if ind1 != ind2 && ! in(ind1 + 1, changed) && ! in(ind2 + 1, changed)

				ind1 += 1
				ind2 += 1

				pCopy = 1.0 / (1.0 + exp( (- w) * (roundPayoffs[ind2] - roundPayoffs[ind1])))
				if rand() < pCopy

					if rand() < imitationCoupling
						population[ind1].normNumber = population[ind2].normNumber
					else
						if rand() < 0.5
							newNum = (population[ind1].normNumber & 15) + (population[ind2].normNumber & (255 - 15))
						else
							newNum = (population[ind2].normNumber & 15) + (population[ind1].normNumber & (255 - 15))
						end

						population[ind1].normNumber = population[ind2].normNumber
					end

					# population[ind1].normNumber = population[ind2].normNumber
					push!(changed, ind1)
				end
			end
		end


		#random drift applied uniformly to all individuals
		for j in 1:NUMAGENTS
			if rand() < ustrat
				num = trunc(Int, ceil(rand() * length(NORMS)))
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
	println("Computing image matrix")
	mostRecentImgMatrix = imageMatrix(reputations, population)

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

