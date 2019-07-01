# ---------------- Imports and top level parameters --------------------
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools as itr

DEBUG = False
PROGRESSVERBOSE = True


# ----------------  Simulation parameters  ----------------

NUMAGENTS = 100

NUMGENERATIONS = 1500

NORM = np.array([[1, 0], [1, 1]])

#observation error
Eobs = 0.0

#cooperation error
Ecoop = 0.0

#imitation pressure
w = 1.0

#random strategy adoption
ustrat = 0.00

#random type mutation 0 --> 1
u01 = 0.00

#random type mutation 1 --> 0
u10 = 0.00

#cooperation benefit
gameBenefit = 0.0

#cooperation cost
gameCost = -2


# --------------- Simulation Classes ---------------------

class DonationGame:
	def __init__(self, benefit, cost):
		self.b = benefit
		self.c = cost

		#Format is for matrix being the row players payoff,
		# where 0 stands for Defect, 1 for Cooperate
		self.payoffMatrix = np.array([[0.0, benefit],[-cost, benefit - cost]])

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
		if strat is not None:
			self.strategy = strat
		else:
			self.strategy = ALLC

		self.reputations = [1 for i in range(NUMAGENTS)]

def imitationUpdate(population, payoffs):
	individuals = np.random.choice(range(NUMAGENTS), size = 2)

	ind1 = individuals[0]
	ind2 = individuals[1]

	pCopy = 1.0 / ( 1 + np.exp( - w * (payoffs[ind2] - payoffs[ind1])))

	if np.random.random() < pCopy:
		if PROGRESSVERBOSE:
			print(" --- social imitation occuring in generation: " + str(i))

		if population[ind1].type == population[ind2].type:
			population[ind1].strategy = population[ind2].strategy
		else:
			population[ind1].strategy = np.flip(population[ind2].strategy) #************** Important Assumption

			#switching types via imitation
			#population[ind1].type = population[ind2].type


	for j in range(len(population)):
		if np.random.random() < ustrat:
			if PROGRESSVERBOSE:
				print("------ random mutation strategy drift occured to individual: " + str(j))
			np.random.shuffle(STRATEGIES)
			newstrat = STRATEGIES[0]
			population[j].strategy = newstrat

		if population[j].type:
			if np.random.random() < u10:
				population[j].type = 0
		else:
			if np.random.random() < u01:
				population[j].type = 1


	return population



class populationStatistics:
	def __init__(self):
		self.statisticsList = [ None for k in range(NUMGENERATIONS)]

	def generateProportionStatistics(self, population, generation):

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

		if self.statisticsList[generation] is not None:
			self.statisticsList[generation]["proportions"] = statistics

		else:
			self.statisticsList[generation] = {}
			self.statisticsList[generation]["proportions"] = statistics

	def plotTypes(self):
		type0popSequence = [stat["proportions"]["type0"]["total"] for stat in statistics.statisticsList]
		type1popSequence = [stat["proportions"]["type1"]["total"] for stat in statistics.statisticsList]

		plt.plot(type0popSequence, 'bo')
		plt.plot(type1popSequence, "g-")
		plt.show()

	def plotComprehensive(self):
		type0coopFreqs = []
		type1coopFreqs = []

		type0AllCFreq = []
		type0DiscFreq = []
		type0AllDFreq = []

		type1AllCFreq = []
		type1DiscFreq = []
		type1AllDFreq = []

		for entry in self.statisticsList:

			type0AllCFreq.append(1.0 * entry["proportions"]["type0"]["ALLC"] / entry["proportions"]["type0"]["total"])
			type0DiscFreq.append(1.0 * entry["proportions"]["type0"]["DISC"] / entry["proportions"]["type0"]["total"])
			type0AllDFreq.append(1.0 * entry["proportions"]["type0"]["ALLD"] / entry["proportions"]["type0"]["total"])

			type0coopFreqs.append(type0AllCFreq[-1] + 0.5 * type0DiscFreq[-1])

			

			type1AllCFreq.append(1.0 * entry["proportions"]["type1"]["ALLC"] / entry["proportions"]["type1"]["total"])
			type1DiscFreq.append(1.0 * entry["proportions"]["type1"]["DISC"] / entry["proportions"]["type1"]["total"])
			type1AllDFreq.append(1.0 * entry["proportions"]["type1"]["ALLD"] / entry["proportions"]["type1"]["total"])

			type1coopFreqs.append(type1AllCFreq[-1] + 0.5 * type1DiscFreq[-1])

		totalCoopFreq = [(type0coopFreqs[i] + type1coopFreqs[i]) / 2.0 for i in range(len(self.statisticsList))]


		numPlotCols = 3
		numPlotRows = 1

		plt.subplot(numPlotCols,numPlotRows,1)

		plt.plot(type0coopFreqs, 'b-')
		plt.plot(type1coopFreqs, "g-")
		plt.plot([(a + b) / 2.0 for a,b in zip(type1coopFreqs, type0coopFreqs)], color = "grey", linestyle= "dashed")
		plt.title("Cooperation Rate by Type")


		
		AllCFreq = [(a + b) / 2.0 for a,b in zip(type1AllCFreq, type0AllCFreq)]
		DiscFreq = [(a + b) / 2.0 for a,b in zip(type1DiscFreq, type0DiscFreq)]
		AllDFreq = [(a + b) / 2.0 for a,b in zip(type1AllDFreq, type0AllDFreq)]


		plt.subplot(numPlotCols,numPlotRows, 2)
		plt.plot(type0AllCFreq, "g-")
		plt.plot(type0DiscFreq, "y-")
		plt.plot(type0AllDFreq, "r-")

		plt.plot(AllCFreq, color = "lightgreen", linestyle = "dashed")
		plt.plot(DiscFreq, color = "palegoldenrod", linestyle = "dashed")
		plt.plot(AllDFreq, color = "lightcoral", linestyle = "dashed")

		plt.title("Type A Strategy Frequencies")



		plt.subplot(numPlotCols,numPlotRows, 3)
		plt.plot(type1AllCFreq, "g-")
		plt.plot(type1DiscFreq, "y-")
		plt.plot(type1AllDFreq, "r-")

		plt.plot(AllCFreq, color = "lightgreen", linestyle = "dashed")
		plt.plot(DiscFreq, color = "palegoldenrod", linestyle = "dashed")
		plt.plot(AllDFreq, color = "lightcoral", linestyle = "dashed")

		plt.title("Type B Strategy Frequencies")



		# plt.subplot(numPlots, 1 , 2)
		# plt.plot([(a + b) / 2.0 for a,b in itr.izip(type1AllCFreq, type0AllCFreq)], "g-")
		# plt.plot([(a + b) / 2.0 for a,b in itr.izip(type1DiscFreq, type0DiscFreq)], "y-")
		# plt.plot([(a + b) / 2.0 for a,b in itr.izip(type1AllDFreq, type0AllDFreq)], "r-")
		# plt.plot("Whole Population Strategy Frequencies")


		plt.show()
	


# ------------------- Simulation Initialization ----------------

game = DonationGame(gameBenefit, gameCost)

population = [Agent(int(np.floor(i / 50.0)), i, strat = STRATEGIES[i % 3]) for i in range(NUMAGENTS)]

judgeCycles = list(range(len(population))) * 2

statistics = populationStatistics()

# -------------------- Simulation Helper Function Declarations -----------------

def updateReputations(pop, reputationUpdates, generation):
	for i in range(len(pop)):

		for j in range(len(reputationUpdates)):
			if reputationUpdates[i][j] is None:
				reputationUpdates[i][j] = pop[i].reputations[j]


		pop[i].reputations = reputationUpdates[i]

	return pop


# ------------------- Generation Loop ------------------------

for i in range(NUMGENERATIONS):
	roundPayoffs = np.zeros(NUMAGENTS)

	reputationUpdates = [[None for x in range(NUMAGENTS)] for y in range(NUMAGENTS)]

	np.random.shuffle(population)

	for j, agent in enumerate(population):

		for k, adversary in enumerate(population[j:]):
			agentRep = adversary.reputations[agent.ID]
			adversaryRep = agent.reputations[adversary.ID]

			if DEBUG:
				print(" ------ agentRep, adversaryRep: " + str(agentRep) + " , " + str(adversaryRep))
				if type(agentRep) != int:
					print("------------ adversary reputation list: " + str(adversary.reputations))
					print("------------ agentRep type: " + str(type(agentRep)))

			agentAction = agent.strategy[adversaryRep]
			adversaryAction = adversary.strategy[agentRep]

			if DEBUG:
				print(" ------ agent strategy, adversary strategy: " + str(agent.strategy) + " , " + str(adversary.strategy))

			if DEBUG:
				print(" ------ agent Action, adversary action:  " + str(agentAction) + " , " + str(adversaryAction))
		
			agentAction = game.moveError(agentAction)
			adversaryAction = game.moveError(adversaryAction)

			if DEBUG:
				print(" ------ errored agent Action, adversary action:  " + str(agentAction) + " , " + str(adversaryAction))
			
			agentPayoff, adversaryPayoff = game.play(agentAction, adversaryAction)

			if DEBUG:
				print(" ------ Agent Payoff: " + str(agentPayoff))

			roundPayoffs[j] += agentPayoff
			roundPayoffs[k] += adversaryPayoff

			#judgement by judger

			judgeNumber = judgeCycles[len(population) + (j - 1) - (k + 1)] 
			judge = population[judgeNumber]

			if np.random.random() < judge.empathy[agent.type]:
				newrep = int(NORM[agentAction, adversaryRep])

				if DEBUG:
					print("------ Empathetic Judgement occured")

			else:
				if DEBUG:
					print("------ trying to set newrep to " + str(NORM[agentAction, judge.reputations[k]]))
					print("------ judge's view of adversary reputation is: " + str(judge.reputations[adversary.ID]))
				newrep = int(NORM[agentAction, judge.reputations[adversary.ID]])

			reputationUpdates[judgeNumber][agent.ID] = newrep


	population = updateReputations(population, reputationUpdates, i)
	if PROGRESSVERBOSE:
		print("--- updated reputations for generation: " + str(i))

	#now for strategy updating via social contagion

	population = imitationUpdate(population, roundPayoffs)

	statistics.generateProportionStatistics(population, i)

	if PROGRESSVERBOSE:
		print("**Completed Generation: " + str(i))


# ------------------- Simulation Statistics Analysis ---------------------

statistics.plotComprehensive()





