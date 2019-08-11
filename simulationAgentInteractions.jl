

function moveError(move, ec)
	if move == 0
		return 0
	elseif rand() < ec
		return 0
	else 
		return 1
	end
end

function updateReps!(reputations, population, a,b,action, perpetratorNorms, relativeNorms, uvisibility, NORMS)
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

function batchUpdate!(numagents,batchsize, reputations, population, perpetratorNorms, relativeNorms, uvisibility, oneShotMatrix, roundPayoffs, cooperationRate, NORMS)
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
			
			updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility, NORMS)
		end

		cooperationRate += action

	end

	updateReps!(reputations,population, a, b, action, perpetratorNorms, relativeNorms, uvisibility, NORMS)

	return cooperationRate
end
