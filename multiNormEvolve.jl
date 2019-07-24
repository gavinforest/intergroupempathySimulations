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