import numpy as np


NUMAGENTS = 100

NUMGENERATIONS = 1500

NORM = np.array([[]])

#observation error
Eobs = 0.0

#cooperation error
Ecoop = 0.0

#imitation pressure
w = 1.0

#random strategy adoption
u = 0.0025

class DonationGame:
	def __init__(self, benefit, cost):
		self.b = benefit
		self.c = cost
		self.payoffMatrix = np.array([[0, benefit],[-cost, benefit - cost]])

	def moveError(self, strat):
		if np.random.random() < Ecoop and strat:
			return 0
		else:
			return strat

	def play(self, strat1, strat2):
		return self.payoffMatrix[strat1,strat2], self.payoffMatrix[strat2, strat1]

class Agent:
	def __init__(self, AgentType, ID, empathy0 = 0.0, empathy1 = 0.0, strat = None):
		self.type = AgentType
		self.ID = ID
		self.empathy = [empathy0, empathy1]
		if self.strategy != None:
			self.strategy = strat
		else:
			self.strategy = np.array([1,1])

		self.reputations = [1 for i in range(NUMAGENTS)]



def updateReputations(pop, reputationUpdates):
	for i in range(len(pop)):

		for j in range(len(reputationUpdates)):
			if reputationUpdates[j] is None:
				reputationUpdates[j] = pop[i].reputations[j]


		pop[i].reputations = reputationUpdates

	return pop

game = DonationGame(1.3, 1)

ALLC = np.array([1,1])
DISC = np.array([0,1])
ALLD = np.array([0,0])

STRATEGIES = [ALLC, DISC, ALLD]



population = [Agent(int(np.floor(i / 50.0)), i) for i in range(NUMAGENTS)]

for i in range(NUMGENERATIONS):
	roundPayoffs = np.zeros(NUMAGENTS)

	reputationUpdates = [[None for x in range(NUMAGENTS)] for y in range(NUMAGENTS)]

	for j, agent in enumerate(population):

		judgers = [l for l in range(NUMAGENTS) if l != j]
		np.random.shuffle(judgers)

		count = 0

		for k, adversary in enumerate(population):
			if j != k:
				agentRep = adversary.reputations[j]
				adversaryRep = agent.reputations[k]

				agentAction = agent.strategy[adversaryRep]
				adversaryAction = adversary.strategy[agentRep]

			
				agentAction = game.moveError(agentAction)
				adversaryAction = game.moveError(adversaryAction)

				
				agentPayoff, adversaryPayoff = game.play(agentAction, adversaryAction)

				roundPayoffs[j] += agentPayoff
				roundPayoffs[k] += adversaryPayoff

				#judgement by judger

				judgeNumber = judgers[count]
				judge = population[judgeNumber]

				if np.random.random() < judge.empathy[agent.type]:
					newrep = NORM[agentAction, adversaryRep]

				else:
					newrep = NORM[agentAction, judge.reputations[k]]

				reputationUpdates[judgeNumber][j] = newrep

				count += 1


	population = updateReputations(population, reputationUpdates)

	#now for strategy updating via social contagion

	individuals = np.random.choice(range(NUMAGENTS), size = 2)

	ind1 = individuals[0]
	ind2 = individuals[1]

	pCopy = 1.0 / ( 1 + np.exp( - w * (roundPayoffs[ind2] - roundPayoffs[ind1])))

	if np.random.random() < pCopy:
		if population[ind1].type == population[ind2].type:
			population[ind1].strategy = population[ind2].strategy
		else:
			population[ind1].strategy = np.flip(population[ind2].strategy)


	for i in range(len(population)):
		if np.random.random() < u:
			np.random.shuffle(STRATEGIES)
			newstrat = STRATEGIES[0]

	















