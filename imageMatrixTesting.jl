import LinearAlgebra


DEBUG = true
PROGRESSVERBOSE = true
TEST = false



const SS = [[1, 0], [1, 1]]
const SJ = [[1, 0], [0, 1]]
const SC = [[0, 0], [1, 1]]
const SH = [[0, 0], [0, 1]]

# const NORMS = [SC, SH, SJ, SS]
const STRATNUMS = collect(1:4)
const STRATNAMES = ["SC", "SH", "SJ", "SS"]

function genNorm(num)
	nums = digits(num, base = 2, pad = 4)
	mat = LinearAlgebra.zeros(Int, 2,2)
	mat[1,1] = nums[1]
	mat[2,2] = nums[4]
	mat[2,1] = nums[3]
	mat[1,2] = nums[2]

	# mat = [[nums[4], nums[3]],[nums[2], nums[1]]]
	return mat
end

# mutable struct Agent 
# 	type::Int
# 	ID::Int
# 	norms::Tuple{Vararg{Array{Int,2},4}}
# 	normNumber::Int
# end

mutable struct Agent 
	type::Int
	ID::Int
	norm
	normNumber::Int
end

function makeAgent(type, ID, normNumber)
	# numString = base(4,normNumber, 4)
	# norms = [NORMS[parse(Int,s)] for s in numString]
	# norms = tuple([NORMS[normNumber] for i in 1:4]...)
	myAgent = Agent(type, ID, genNorm(normNumber), normNumber)
	return myAgent
end


# function generateStatistics!(statList,generation, population, cooperationRate)
# 	typeProportions = LinearAlgebra.zeros(Float64, 256)
	

# 	for j in 1:length(population)
# 		typeProportions[population[j].normNumber + 1] += 1

# 	end
# 	typeProportions = typeProportions / length(population)


# 	statList[generation] = (typeProportions, cooperationRate)
# 	return statList

# end


function interactionIterate()
	PROGRESSVERBOSE = true

	NUMAGENTSPERNORM = 200
	NUMAGENTS = 16 * NUMAGENTSPERNORM
	# NUMGENERATIONS = 100000
	INTERACTIONSPERAGENT = 100

	# Eobs = 0.02
	# Ecoop = 0.02
	# w = 1.0
	# ustrat = 0.0025
	# u01 = 0.0 #type mutation rate 0 -> 1
	# u10 = 0.0 #type mutation rate 1 -> 0
	# gameBenefit = 5.0
	# gameCost = 1.0
	# intergroupUpdateP = 0.0


	# oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
	population = [makeAgent(i % 2, i, (i-1) % 16) for i in 1:NUMAGENTS]
	# arguments are: type, ID, strategy number, empathy

	# println("empathy matrix: $empathyMatrix")

	# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	reputations = LinearAlgebra.ones(Int, NUMAGENTS, NUMAGENTS)

	# statistics = [ (LinearAlgebra.zeros(Float64, 256), 0.0) for i in 1:NUMGENERATIONS]
	# statistics::Array{Tuple{Array{Float64,1}, Float64},1}


	# for i in 1:NUMGENERATIONS
	startTime = time_ns()
	# roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)
	# reputationUpdates = LinearAlgebra.zeros(Int, NUMAGENTS, NUMAGENTS)

	# cooperationRate = 0.0
	# cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

	for i in 1:(NUMAGENTS * INTERACTIONSPERAGENT)
		#judges

		# for i in 1:INTERACTIONSPERAGENT

			#agents

		a = trunc(Int, ceil(rand() * NUMAGENTS))

		adversaryID = trunc(Int, ceil(rand() * NUMAGENTS))
		#randomly selected but agents never have to play themselves.

		adversaryRep = reputations[a, adversaryID]

		agentAction = adversaryRep + 1
		# agentAction = moveError(agentAction, Ecoop)

		# agentPayoff, adversaryPayoff = oneShotMatrix[agentAction + 1]

		# roundPayoffs[agentID] += agentPayoff
		# roundPayoffs[adversaryID] += adversaryPayoff

		for j in 1:NUMAGENTS

			# judgetype = population[j].type
			
			judgesview = reputations[j,adversaryID]

			# normInd = 2 * population[agentID].type + population[adversaryID].type

			# newrep = population[j].norms[normInd][agentAction, judgesview]

			newrep = population[j].norm[agentAction, judgesview + 1]
			newrep::Int

			# if rand()<Eobs
			# 	if newrep ==1
			# 		newrep = 0
			# 	else
			# 		newrep = 1
			# 	end
			# end


			# reputationUpdates[j,a] = newrep
			reputations[j,a] = newrep

		end

		if i % 1000 == 0
			println("Simulated $i interactions")
		end

				#strategy specific coop rate calculation
				# cooperationRate += agentAction


	end


	# reputations = reputationUpdates

	println("finished loops, computing image matrix now")

	imageMatrix = LinearAlgebra.zeros(Float64, 16,16)

	for i in 1:16
		for j in 1:16
			avg = 0.0
			avgDenom = 0.0

			for k in range(i, step = 16, stop = NUMAGENTS)
				for n in range(j, step = 16, stop = NUMAGENTS)
					avg += reputations[k,n]
					avgDenom += 1.0
				end
			end

			imageMatrix[i,j] = avg/avgDenom

		end
	end
	endTime = time_ns()
	elapsedSecs = (endTime - startTime) / 1.0e9

	if PROGRESSVERBOSE
		println("Completed simulation in $elapsedSecs seconds")
		# println("statistics for this generration are: $(statistics[i])")
	end

	agreementMatrix = LinearAlgebra.zeros(Int, NUMNORMS, NUMAGENTS)

	

	return imageMatrix






	# cooperationRate = cooperationRate / cooperationRateDenominator

	# statistics = generateStatistics!(statistics, generation, population, cooperationRate)

	# for j in 1:NUMAGENTS
	# 	for a in 1:INTERACTIONSPERAGENT
	# 		reputations[j, real(reputationUpdates[j,a])] = imag(reputationUpdates[j,a])


	#imitation update. intergroupUpdateP calculations rely on parity of indices originating in original
	#creation of population. Beware changing that.
	# ind1 = rand(0:(NUMAGENTS-1))
	# ind2 = (ind1 + rand(1:(NUMAGENTS-1))) % NUMAGENTS

	# inter = rand() < intergroupUpdateP

	# if (ind2 - ind1) %2 == 0 && inter
	# 	ind2 = (ind2 + 1) % NUMAGENTS
	# elseif (ind2 - ind1) %2 == 1 && !inter
	# 	ind2 = (ind2 + 1) % NUMAGENTS
	# 	if ind1 == ind2
	# 		ind2 = (ind2 + 2) % NUMAGENTS
	# 	end
	# end
		
	# ind1 += 1
	# ind2 += 1



	# pCopy = 1.0 / (1.0 + exp( (- w) * (roundPayoffs[ind2] - roundPayoffs[ind1])))
	# if rand() < pCopy
	# 	population[ind1].strategy = population[ind2].strategy
	# 	population[ind1].stratString = population[ind2].stratString
	# 	population[ind1].stratNumber = population[ind2].stratNumber
	# end

	# if PROGRESSVERBOSE && i%1000==0 && DEBUG
	# 	println("--- simulated generation")
	# end

	# #random drift applied uniformly to all individuals
	# stratNames = ["ALLD","DISC","ALLC"]
	# for j in 1:NUMAGENTS
	# 	if rand() < ustrat
	# 		num = rand(STRATNUMS)
	# 		population[j].stratNumber = num
	# 		population[j].strategy = STRATEGIES[num]
	# 		population[j].stratString = STRATNAMES[num]
	# 	end

	# 	if population[j].type == 0 && u01 > rand()
	# 		population[j].type = 1
	# 	elseif population[j].type == 1 && u10 > rand()
	# 		population[j].type = 0
	# 	end
	# end
end

function interactionIterateCondensed()
	PROGRESSVERBOSE = true

	NORMS = [genNorm(i) for i in 0:15]

	NUMAGENTSPERNORM = 200
	NUMNORMS = 16
	NUMAGENTS = NUMNORMS * NUMAGENTSPERNORM
	# NUMGENERATIONS = 100000
	INTERACTIONSPERAGENT = 100

	


	# oneShotMatrix = [(0.0, 0.0),(- gameCost, gameBenefit - gameCost)]


	# intergroupUpdateP relies on the alternation of types from the first argument. Change with care.
	population = [makeAgent(i % 2, i, (i-1) % 16) for i in 1:NUMAGENTS]
	# arguments are: type, ID, strategy number, empathy

	# println("empathy matrix: $empathyMatrix")

	# reputations = rand([0,1], NUMAGENTS, NUMAGENTS)
	reputations = LinearAlgebra.ones(Int, NUMNORMS, NUMAGENTS)

	# statistics = [ (LinearAlgebra.zeros(Float64, 256), 0.0) for i in 1:NUMGENERATIONS]
	# statistics::Array{Tuple{Array{Float64,1}, Float64},1}


	# for i in 1:NUMGENERATIONS
	startTime = time_ns()
	# roundPayoffs = LinearAlgebra.zeros(Float64, NUMAGENTS)
	# reputationUpdates = LinearAlgebra.zeros(Int, NUMAGENTS, NUMAGENTS)

	# cooperationRate = 0.0
	# cooperationRateDenominator = NUMAGENTS * INTERACTIONSPERAGENT

	for i in 1:(NUMAGENTS * INTERACTIONSPERAGENT)
		#judges

		# for i in 1:INTERACTIONSPERAGENT

			#agents

		a = trunc(Int, ceil(rand() * NUMAGENTS))

		adversaryID = trunc(Int, ceil(rand() * NUMAGENTS))
		#randomly selected but agents never have to play themselves.

		adversaryRep = reputations[population[a].normNumber + 1, adversaryID]

		agentAction = adversaryRep + 1
		# agentAction = moveError(agentAction, Ecoop)

		# agentPayoff, adversaryPayoff = oneShotMatrix[agentAction + 1]

		# roundPayoffs[agentID] += agentPayoff
		# roundPayoffs[adversaryID] += adversaryPayoff

		for j in 1:NUMNORMS

			# judgetype = population[j].type
			
			judgesview = reputations[j,adversaryID]

			# normInd = 2 * population[agentID].type + population[adversaryID].type

			# newrep = population[j].norms[normInd][agentAction, judgesview]

			newrep = NORMS[j][agentAction, judgesview + 1]
			newrep::Int

			# if rand()<Eobs
			# 	if newrep ==1
			# 		newrep = 0
			# 	else
			# 		newrep = 1
			# 	end
			# end


			# reputationUpdates[j,a] = newrep
			reputations[j,a] = newrep

		end

		if i % 1000 == 0
			println("Simulated $i interactions")
		end

				#strategy specific coop rate calculation
				# cooperationRate += agentAction


	end


	# reputations = reputationUpdates

	println("finished loops, computing image matrix now")

	imageMatrix = LinearAlgebra.zeros(Float64, 16,16)

	for i in 1:16
		for j in 1:16
			avg = 0.0
			avgDenom = 0.0

			for n in range(j, step = 16, stop = NUMAGENTS)
				avg += reputations[i,n]
				avgDenom += 1.0
			end

			imageMatrix[i,j] = avg/avgDenom

		end
	end
	endTime = time_ns()
	elapsedSecs = (endTime - startTime) / 1.0e9

	if PROGRESSVERBOSE
		println("Completed simulation in $elapsedSecs seconds")
		# println("statistics for this generration are: $(statistics[i])")
	end

	return imageMatrix

end


norms = [genNorm(i) for i in 0:15]

function probProduct(x,s,i,j,k)
	return x[k] * transpose([1 - s[j,k], s[j,k]]) * norms[(i-1) % 16 + 1] * [1 - s[i,k], s[i,k]]
end


function iterateMatrix(x,s)

	# s = rand([0,1],16,16)

	width = size(s)[1]

	sp = LinearAlgebra.ones(Float64, size(s)[1],size(s)[2])

	for i in 1:width
		for j in 1:width
			sp[i,j] = sum(map(y -> probProduct(x,s,i,j,y), 1:width))
		end
	end

	return sp
end

function bound(m)
	for i in 1:size(m)[1]
		for j in 1:size(m)[2]
			if m[i,j] > 1.0
				m[i,j] = 1.0
			elseif m[i,j] < 0.0
				m[i,j] = 0.0
			end
		end
	end
	return m
end



function interactionMatrix()

	epsilon = 0.02

	numiters = 1000

	x = LinearAlgebra.ones(Float64, 16) / 16.0

	s = LinearAlgebra.ones(Float64, 16,16)
	# s = rand([0,1], 16,16)

	saved = [LinearAlgebra.zeros(Float64,1,1) for i in 1:numiters]

	for i in 1:numiters
		sp = iterateMatrix(x,s)
		s = bound((1 - epsilon) * s + epsilon * sp)
		# s = copy(sp)
		saved[i] = copy(s)

		if i % 10 == 0
			println("Simulated iteration $i")
		end
	end
	print(typeof(saved))
	return saved
end

function semiagentInteractionMatrix()
	# epsilon = 0.01
	duplicates = 15
	numiters = 100

	x = LinearAlgebra.ones(Float64, duplicates * 16) / (duplicates * 16.0)

	s = LinearAlgebra.ones(Float64, duplicates * 16, duplicates * 16)

	saved = [LinearAlgebra.zeros(Float64,1,1) for i in 1:numiters]

	for i in 1:numiters
		sp = iterateMatrix(x,s)
		s = sp
		saved[i] = copy(s)
		if i% 10 == 0
			println("Simulated iteration $i")
		end
	end


	return saved

end





# function testEvolve()
	

# 	if TEST
# 		stats = evolve(testPopParams, testEnvParams, testNorm, LinearAlgebra.zeros(Float64, 2,2))
# 		println(stats[end])
# 	else 
# 		return evolve(testPopParams, testEnvParams, testNorm, LinearAlgebra.zeros(Float64, 2,2))
# 	end

# end
# defaultPopParams = Dict("numAgents" => 100, "numGenerations" => 150000)
# defaultEnvParams = Dict("Ecoop" => 0.0, "Eobs" => 0.0, "ustrat" => 0.001, "u01" => 0.0, "u10" => 0.0, "w" => 1.0,
# 					"gameBenefit" => 5.0, "gameCost" => 1.0, )

# defaultNorm = LinearAlgebra.ones(Int,2,2)
# defaultNorm[1,2] = 0


# if TEST
# 	testEvolve()
# end

