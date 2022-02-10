#assesment rules
#Format: First index 1-> good donor, 2-> bad donor
#		Second index 1->cooperates,  2-> defects
#        Third index 1->good recipient,  2-> bad recipient

L1a = [[[1,1], [0,1]], [[1,1], [0,0]]]
L2a = [[[1,0], [0,1]], [[1,1], [0,0]]] #Consistent Standing
L3a = [[[1,1], [0,1]], [[1,1], [0,1]]] #Simple Standing
L4a = [[[1,1], [0,1]], [[1,0], [0,1]]]
L5a = [[[1,0], [0,1]], [[1,1], [0,1]]]
L6a = [[[1,0], [0,1]], [[1,0], [0,1]]] #Stern Judging
L7a = [[[1,1], [0,1]], [[1,0], [0,0]]] #Staying
L8a = [[[1,0], [0,1]], [[1,0], [0,0]]] #Judging
ALLCa = [[[1,1],[1,1]],[[1,1], [1,1]]]
ALLDa = [[[0,0],[0,0]],[[0,0], [0,0]]]

function explainAssessmentRule(rule)
	println("Explaining assessment rule " * string(rule))
	for action in [1,2]
		
		for donorRep in [1,2]

			for recipientRep in [1,2]
				if donorRep == 1
					print("Good ")
				else
					print("Bad ")
				end

				if action == 1
					print(" cooperates ")
				else
					print(" defects")
				end


				if recipientRep == 1
					print(" against Good ")
				else
					print(" against Bad ")
				end

				if rule[donorRep][action][recipientRep] == 1
					print(" --> Good\n")
				else
					print(" --> Bad\n")
				end

			end
		end
	end
end

# explainAssessmentRule(L1a)
# explainAssessmentRule(L2a)
# explainAssessmentRule(L3a)
# explainAssessmentRule(L4a)
# explainAssessmentRule(L5a)
# explainAssessmentRule(L6a)
# explainAssessmentRule(L7a)
# explainAssessmentRule(L8a)

#action rules
#Format: first index 1->good self, 2-> bad self
#		second index 1->good recip, 2->bad recip
#
#content: 1 -> cooperate, 2-> defect

L1b = [[1,2],[1,1]]
L2b = [[1,2],[1,1]]
L3b = [[1,2],[1,2]]
L4b = [[1,2],[1,2]]
L5b = [[1,2],[1,2]]
L6b = [[1,2],[1,2]]
L7b = [[1,2],[1,2]]
L8b = [[1,2],[1,2]]
ALLCb = [[1,1],[1,1]]
ALLDb = [[2,2],[2,2]]

function explainActionRule(rule)
	println("Explaining action rule " * string(rule))
	for actorRep in [1,2]
		for recipientRep in [1,2]
			if actorRep == 1
				print("Good meets")
			else
				print("Bad meets")
			end

			if recipientRep == 1
				print(" Good ")
			else
				print(" Bad ")
			end

			if rule[actorRep][recipientRep] == 1
				print(" --> Cooperate\n")
			else
				print(" --> Defect\n")
			end
		end
	end
end

# explainActionRule(L1b)
# explainActionRule(L2b)
# explainActionRule(L3b)
# explainActionRule(L4b)
# explainActionRule(L5b)
# explainActionRule(L6b)
# explainActionRule(L7b)
# explainActionRule(L8b)



