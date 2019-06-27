# ---------------- Imports and top level parameters --------------------
import numpy as np
import matplotlib.pylpot as plt
import json

DEBUG = True


# ----------------  Simulation parameters  ----------------

NUMAGENTS = 100

NUMGENERATIONS = 1500

NORM = np.array([[1, 0], [0 , 1]])

#observation error
Eobs = 0.0

#cooperation error
Ecoop = 0.0

#imitation pressure
w = 1.0

#random strategy adoption
u = 0.0025


# --------------- Simulation Classes ---------------------

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

ALLC = np.array([1,1])
DISC = np.array([0,1])
ALLD = np.array([0,0])

STRATEGIES = [ALLC, DISC, ALLD]

class Agent:
	def __init__(self, AgentType, ID, empathy0 = 0.0, empathy1 = 0.0, strat = None):
		self.type = AgentType
		self.ID = ID
		self.empathy = [empathy0, empathy1]
		if strat != None:
			self.strategy = strat
		else:
			self.strategy = ALLC

		self.reputations = [1 for i in range(NUMAGENTS)]



def updateReputations(pop, reputationUpdates):
	for i in range(len(pop)):

		for j in range(len(reputationUpdates)):
			if reputationUpdates[j] is None:
				reputationUpdates[j] = pop[i].reputations[j]


		pop[i].reputations = reputationUpdates

	return pop

class populationStatistics:
	def __init__(self):
		self.statisticsList = []



	def generateStatistics(population, generation):

		type0 = [agent for agent in population if agent.type == 0]
		type1 = [agent for agent in population if agent.type == 1]

		TotalType0 = len(type0)
		TotalType1 = len(type1)


		ALLCType0 =0
		DISCType0 = 0
		ALLDType0 = 0

		for agent in type0:
			if np.array_equal(agent.strategy, ALLC):
				ALLCType0 += 1
			elif np.array_equal(agent.strategy, ALLD):
				ALLDType0 += 1
			else:
				DISCType0 += 1


		ALLCType1 = 0
		DISCType1 = 0
		ALLDType1 = 0

		for agent in type1:
			if np.array_equal(agent.strategy, ALLC):
				ALLCType1 +=1
			elif np.array_equal(agent.strategy, ALLD):
				ALLDType1 += 1
			else:
				DISCType1 += 1

		statistics = {"generation": generation, "type0": {"total": TotalType0, "ALLC": ALLCType0, "DISC": DISCType0, "ALLD": ALLDType0},
												"type1": {"total": TotalType1, "ALLC": ALLCType1, "DISC": DISCType1, "ALLD": ALLDType1}}

		self.statisticsList.append(statistics)

	


# ------------------- Simulation Initialization ----------------

game = DonationGame(1.3, 1)

population = [Agent(int(np.floor(i / 50.0)), i, strat = STRATEGIES[i % 3]) for i in range(NUMAGENTS)]

statistics = populationStatistics()

# ------------------- Generation Loop ------------------------

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

	statistics.generateStatistics(population, i)

	if DEBUG:
		print("Completed Generation: " + str(i))


# ------------------- Simulation Statistics Analysis ---------------------

type0popSequence = [stat["type0"]["total"] for stat in statistics.statisticsList]
type1popSequence = [stat["type1"]["total"] for stat in statistics.statisticsList]







