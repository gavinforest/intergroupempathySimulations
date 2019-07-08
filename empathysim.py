# ---------------- Imports and top level parameters --------------------
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools as itr
from multiprocessing import Pool
import time
import random

DEBUG = False
PROGRESSVERBOSE = True


# ----------------  Simulation parameters  ----------------

NUMAGENTS = 100

NUMGENERATIONS = 1500

NUMPOPULATIONS = 1

NORM = np.array([[1, 0], [0, 1]], dtype="int64")

#observation error
Eobs = 0.0

#cooperation error
Ecoop = 0.0

#imitation pressure
w = 1.0

#random strategy adoption
ustrat = 0.001

#random type mutation 0 --> 1
u01 = 0.00

#random type mutation 1 --> 0
u10 = 0.00

#cooperation benefit
gameBenefit = 5

#cooperation cost
gameCost = 1


# --------------- Simulation Classes ---------------------

class DonationGame:
	def __init__(self, benefit, cost):
		self.b = benefit
		self.c = cost

		#Format is for matrix being the row players payoff,
		# where 0 stands for Defect, 1 for Cooperate
		self.payoffMatrix = np.array([[0.0, benefit],[-cost, benefit - cost]])
		self.oneshot = [(0.0,0.0), (-cost, benefit - cost)]

	def moveError(self, strat):
		if random.random() < Ecoop and strat:
			return 0
		else:
			return strat

	def play(self, strat1, strat2):
		return self.payoffMatrix[strat1,strat2], self.payoffMatrix[strat2, strat1]

ALLC = np.array([1,1])
DISC = np.array([0,1])
ALLD = np.array([0,0])

STRATEGIES = [ALLC, DISC, ALLD]

def stratToString(strategy):
	if np.array_equal(strategy, ALLC):
		return "ALLC"
	elif np.array_equal(strategy, DISC):
		return "DISC"
	elif np.array_equal(strategy, ALLD):
		return "ALLD"
	else:
		if DEBUG:
			print("Got strange strategy: " + str(strategy))
		return "ERROR"

def stratNumber(strategy):
	string = stratToString(strategy)
	if string == "ALLC":
		return 0
	elif string == "DISC":
		return 1
	else:
		return 2

class Agent:
	def __init__(self, AgentType, ID, empathy0 = 0.0, empathy1 = 0.0, strat = None):
		self.type = AgentType
		self.ID = ID
		self.empathy = [empathy0, empathy1]
		if strat is not None:
			self.strategy = strat
		else:
			self.strategy = ALLC

		self.stratString = stratToString(self.strategy)
		self.stratNumber = stratNumber(self.strategy)

		# self.reputations = np.random.randint(2, size = NUMAGENTS)

def imitationUpdate(population, payoffs, generation):
	individuals = np.random.choice(range(NUMAGENTS), size = 2)

	ind1 = individuals[0]
	ind2 = individuals[1]

	pCopy = 1.0 / ( 1 + np.exp( - w * (payoffs[ind2] - payoffs[ind1])))

	if random.random() < pCopy:
		if PROGRESSVERBOSE:
			print(" --- social imitation occuring in generation: " + str(generation))


		population[ind1].strategy = population[ind2].strategy
		# if population[ind1].type == population[ind2].type:
		# 	population[ind1].strategy = population[ind2].strategy
		# else:

			# population[ind1].strategy = np.flip(population[ind2].strategy) Reputations based Gavin!


			#switching types via imitation
			#population[ind1].type = population[ind2].type


	for j in range(len(population)):
		if random.random() < ustrat:
			if PROGRESSVERBOSE:
				print("------ random mutation strategy drift occured to individual: " + str(j))
			np.random.shuffle(STRATEGIES)
			newstrat = STRATEGIES[0]
			population[j].strategy = newstrat

		if population[j].type:
			if random.random() < u10:
				population[j].type = 0
		else:
			if random.random() < u01:
				population[j].type = 1


	return population



class populationStatistics:
	def __init__(self):
		self.statisticsList = [ None for k in range(NUMGENERATIONS)]
		self.populationHistory = [None for k in range(NUMGENERATIONS)]

	def generateStatistics(self, population, reputations, generation, cooperationRateD):

		if DEBUG:
			print("--- generating statistics for generation: " + str(generation))

		# self.populationHistory[generation] = population[:]



		#TYPE PROPORTION STATISTICS

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

		statistics = {"type0": {"total": TotalType0, "ALLC": ALLCType0, "DISC": DISCType0, "ALLD": ALLDType0},
												"type1": {"total": TotalType1, "ALLC": ALLCType1, "DISC": DISCType1, "ALLD": ALLDType1}}

		if self.statisticsList[generation] is not None:
			self.statisticsList[generation]["proportions"] = statistics

		else:
			self.statisticsList[generation] = {}
			self.statisticsList[generation]["proportions"] = statistics


		repStats = {}

		# typeStratPairPointers = [(agent.type, stratToString(agent.strategy)) for agent in population]
		# typeStratPairTargets = [None for i in range(NUMAGENTS)]
		# for agent in population:
		# 	typeStratPairTargets[agent.ID] = (agent.type, stratToString(agent.strategy))

		stratStringsByID = [None for agent in population]
		for agent in population:
			stratStringsByID[agent.ID] = stratToString(agent.strategy)

		repStats["viewsFromTo"] = {}

		strats = ["ALLC", "DISC", "ALLD"]
		for stratType in strats:
			repStats["viewsFromTo"][stratType] = {}

			for strat in strats:
				repStats["viewsFromTo"][stratType][strat] = []
			
			for agent in population:
				for i, strat in enumerate(stratStringsByID):
					repStats["viewsFromTo"][stratType][strat].append(reputations[agent.ID, i])

			for strat in strats:
				repStats["viewsFromTo"][stratType][strat] = np.average(repStats["viewsFromTo"][stratType][strat])

		self.statisticsList[generation]["reputations"] = repStats

		self.statisticsList[generation]["cooperationRate"] = cooperationRateD
			


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

		plt.figure(1)
		numPlotsCol = 3
		numPlotsRow = 1

		plt.subplot(numPlotsCol,numPlotsRow,1)

		plt.plot(type0coopFreqs, 'b-')
		plt.plot(type1coopFreqs, "g-")
		plt.plot([(a + b) / 2.0 for a,b in zip(type1coopFreqs, type0coopFreqs)], color = "grey", linestyle= "dashed")
		plt.title("Cooperation Rate by Type")


		
		AllCFreq = [(a + b) / 2.0 for a,b in zip(type1AllCFreq, type0AllCFreq)]
		DiscFreq = [(a + b) / 2.0 for a,b in zip(type1DiscFreq, type0DiscFreq)]
		AllDFreq = [(a + b) / 2.0 for a,b in zip(type1AllDFreq, type0AllDFreq)]


		plt.subplot(numPlotsCol,numPlotsRow, 2)
		plt.plot(type0AllCFreq, "g-")
		plt.plot(type0DiscFreq, "y-")
		plt.plot(type0AllDFreq, "r-")

		plt.plot(AllCFreq, color = "lightgreen", linestyle = "dashed")
		plt.plot(DiscFreq, color = "palegoldenrod", linestyle = "dashed")
		plt.plot(AllDFreq, color = "lightcoral", linestyle = "dashed")

		plt.title("Type A Strategy Frequencies")



		plt.subplot(numPlotsCol,numPlotsRow, 3)
		plt.plot(type1AllCFreq, "g-")
		plt.plot(type1DiscFreq, "y-")
		plt.plot(type1AllDFreq, "r-")

		plt.plot(AllCFreq, color = "lightgreen", linestyle = "dashed")
		plt.plot(DiscFreq, color = "palegoldenrod", linestyle = "dashed")
		plt.plot(AllDFreq, color = "lightcoral", linestyle = "dashed")

		plt.title("Type B Strategy Frequencies")

		plt.figure(2)


		strats = ["ALLC", "DISC", "ALLD"]

		averageRepALLC = [0 for j in range(NUMGENERATIONS)]
		averageRepALLD = [0 for j in range(NUMGENERATIONS)]		
		averageRepDISC = [0 for j in range(NUMGENERATIONS)]

		for j in range(NUMGENERATIONS):
			averageRepALLC[j] += AllCFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLC"]["ALLC"]
			averageRepALLC[j] += DiscFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["DISC"]["ALLC"]
			averageRepALLC[j] += AllDFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLD"]["ALLC"]

			averageRepALLD[j] += AllCFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLC"]["ALLD"]
			averageRepALLD[j] += DiscFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["DISC"]["ALLD"]
			averageRepALLD[j] += AllDFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLD"]["ALLD"]

			averageRepDISC[j] += AllCFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLC"]["DISC"]
			averageRepDISC[j] += DiscFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["DISC"]["DISC"]
			averageRepDISC[j] += AllDFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLD"]["DISC"]



		for i, strat in enumerate(strats):
			plt.subplot(numPlotsCol, numPlotsRow, i + 1)
			plt.title("Reputations of Strategies as Viewed by " + strat)

			ALLCHist = [self.statisticsList[j]["reputations"]["viewsFromTo"][strat]["ALLC"] for j in range(NUMGENERATIONS)]
			ALLDHist = [self.statisticsList[j]["reputations"]["viewsFromTo"][strat]["ALLD"] for j in range(NUMGENERATIONS)]
			DISCHist = [self.statisticsList[j]["reputations"]["viewsFromTo"][strat]["DISC"] for j in range(NUMGENERATIONS)]

			plt.plot(ALLCHist, "g-")
			plt.plot(DISCHist, "y-")
			plt.plot(ALLDHist, "r-")

			plt.plot(averageRepALLC, color = "lightgreen", linestyle = "dashed")
			plt.plot(averageRepDISC, color = "palegoldenrod", linestyle = "dashed")
			plt.plot(averageRepALLD, color = "lightcoral", linestyle = "dashed")





		plt.show()
	

# -------------------- Simulation Helper Function Declarations -----------------

# def updateReputations(pop, reputationUpdates, generation):

# 	numNones = 0
# 	for i in range(len(pop)):

# 		for j in range(len(reputationUpdates)):
# 			if reputationUpdates[i][j] is None:
# 				reputationUpdates[i][j] = pop[i].reputations[j]
# 				numNones += 1


# 		pop[i].reputations = reputationUpdates[i]

# 	if DEBUG:
# 		print("--- numNones: " + str(numNones))
# 	return pop


# ------------------- Generation Loop Function  ------------------------

def evolve(simulationInstanceID):

	#------- simulation initialization ---------
	game = DonationGame(gameBenefit, gameCost)

	population = [Agent(int(np.floor(i / 50.0)), i, strat = STRATEGIES[i % 2 + 1]) for i in range(NUMAGENTS)]

	reputations = np.random.randint(2, size= (NUMAGENTS, NUMAGENTS))

	statistics = populationStatistics()

	#-------- generation loop---------

	for i in range(NUMGENERATIONS):
		roundPayoffs = np.zeros(NUMAGENTS)

		reputationUpdates = np.zeros((NUMAGENTS, NUMAGENTS))

		cooperationRate = [0.0 for j in range(4)]
		cooperationDenom = [0 for j in range(4)]

		# np.random.shuffle(population)
		startTime = time.time()

		for j in range(NUMAGENTS): #judge

			for k in range(NUMAGENTS): #agent


				# inStart = time.time()

				adversaryID = (k + random.randint(1, NUMAGENTS - 1)) % NUMAGENTS

				adversaryRep = reputations[k, adversaryID]

				if DEBUG:
					print(" ------ agentRep, adversaryRep: " + str(agentRep) + " , " + str(adversaryRep))
					if type(agentRep) != int:
						print("------------ adversary reputation list: " + str(adversary.reputations))
						print("------------ agentRep type: " + str(type(agentRep)))

				agentAction = population[k].strategy[adversaryRep]

				if DEBUG:
					print(" ------ agent strategy, adversary strategy: " + str(agent.strategy) + " , " + str(adversary.strategy))

				if DEBUG:
					print(" ------ agent Action, adversary action:  " + str(agentAction) + " , " + str(adversaryAction))
			
				agentAction = game.moveError(agentAction)


				if DEBUG:
					print(" ------ errored agent Action, adversary action:  " + str(agentAction) + " , " + str(adversaryAction))
				

				agentPayoff, adversaryPayoff = game.oneshot[agentAction]


				if DEBUG:
					print(" ------ Agent Payoff: " + str(agentPayoff))

				roundPayoffs[j] = roundPayoffs[j] + agentPayoff
				roundPayoffs[k] = roundPayoffs[k] + adversaryPayoff

				# if PROGRESSVERBOSE:
				# 	print("------ got past payoffs in: " + str(time.time() - inStart))

				# inStart = time.time()


				#judgement by judger
				oldrep = reputations[j,k]

				if random.random() < population[j].empathy[population[k].type]:
					newrep = NORM[agentAction][adversaryRep]

					# if DEBUG:
					# 	print("------ Empathetic Judgement occured")
				else:
					# if DEBUG:
					# 	print("------ trying to set newrep to " + str(NORM[agentAction, judge.reputations[k]]))
					# 	print("------ judge's view of adversary reputation is: " + str(judge.reputations[adversary.ID]))
					newrep = NORM[agentAction][reputations[j,adversaryID]]


				reputationUpdates[j, k] = newrep - oldrep
				
				# if PROGRESSVERBOSE:
				# 	print("------ got past payoffs and rep in: " + str(time.time() - inStart))

				# inStart = time.time()

				cooperationRate[population[k].stratNumber] += agentAction
				cooperationDenom[population[k].stratNumber] += 1
				cooperationRate[3] += agentAction 
				cooperationDenom[3] += 1

				# if PROGRESSVERBOSE:
				# 	print("------ finished everything else in: " + str(time.time() - inStart))


		#calling here so as to not confuse the population that resulted in this outcome 
		#with the updated population
		if PROGRESSVERBOSE:
			print("--- simulated interactions in: " + str(time.time() - startTime))

		cooperationRateDict = {}
		for j, strat in enumerate(["ALLC", "ALLD", "DISC", "TOTAL"]):
			if cooperationDenom[j] > 0.0:
				cooperationRateDict[strat] = cooperationRate[j] / cooperationDenom[j]

		startTime = time.time()
		statistics.generateStatistics(population, reputations, i, cooperationRateDict)

		if PROGRESSVERBOSE:
			print("--- generated statistics in: " + str(time.time() - startTime))

		startTime = time.time()
		reputations = reputations + reputationUpdates
		# population = updateReputations(population, reputationUpdates, i)
		if PROGRESSVERBOSE:
			print("--- updated reputations for generation: " + str(i))
			print("--- updating reputations took: " + str(time.time() - startTime))

		#now for strategy updating via social contagion

		startTime = time.time()
		population = imitationUpdate(population, roundPayoffs, i)
		if PROGRESSVERBOSE:
			print("--- imitation update took: " + str(time.time() - startTime))
		
		if PROGRESSVERBOSE:
			print("**Completed Generation: " + str(i))

	print("**Completed Simulation: " + str(simulationInstanceID))
	return statistics

#-------------------- Multiple Simulation Analysis Functions -----------------
def plotMultiplePopulations(statLists):
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

	plt.figure(1)
	numPlotsCol = 3
	numPlotsRow = 1

	plt.subplot(numPlotsCol,numPlotsRow,1)

	plt.plot(type0coopFreqs, 'b-')
	plt.plot(type1coopFreqs, "g-")
	plt.plot([(a + b) / 2.0 for a,b in zip(type1coopFreqs, type0coopFreqs)], color = "grey", linestyle= "dashed")
	plt.title("Cooperation Rate by Type")


	
	AllCFreq = [(a + b) / 2.0 for a,b in zip(type1AllCFreq, type0AllCFreq)]
	DiscFreq = [(a + b) / 2.0 for a,b in zip(type1DiscFreq, type0DiscFreq)]
	AllDFreq = [(a + b) / 2.0 for a,b in zip(type1AllDFreq, type0AllDFreq)]


	plt.subplot(numPlotsCol,numPlotsRow, 2)
	plt.plot(type0AllCFreq, "g-")
	plt.plot(type0DiscFreq, "y-")
	plt.plot(type0AllDFreq, "r-")

	plt.plot(AllCFreq, color = "lightgreen", linestyle = "dashed")
	plt.plot(DiscFreq, color = "palegoldenrod", linestyle = "dashed")
	plt.plot(AllDFreq, color = "lightcoral", linestyle = "dashed")

	plt.title("Type A Strategy Frequencies")



	plt.subplot(numPlotsCol,numPlotsRow, 3)
	plt.plot(type1AllCFreq, "g-")
	plt.plot(type1DiscFreq, "y-")
	plt.plot(type1AllDFreq, "r-")

	plt.plot(AllCFreq, color = "lightgreen", linestyle = "dashed")
	plt.plot(DiscFreq, color = "palegoldenrod", linestyle = "dashed")
	plt.plot(AllDFreq, color = "lightcoral", linestyle = "dashed")

	plt.title("Type B Strategy Frequencies")

	plt.figure(2)


	strats = ["ALLC", "DISC", "ALLD"]

	averageRepALLC = [0 for j in range(NUMGENERATIONS)]
	averageRepALLD = [0 for j in range(NUMGENERATIONS)]		
	averageRepDISC = [0 for j in range(NUMGENERATIONS)]

	for j in range(NUMGENERATIONS):
		averageRepALLC[j] += AllCFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLC"]["ALLC"]
		averageRepALLC[j] += DiscFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["DISC"]["ALLC"]
		averageRepALLC[j] += AllDFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLD"]["ALLC"]

		averageRepALLD[j] += AllCFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLC"]["ALLD"]
		averageRepALLD[j] += DiscFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["DISC"]["ALLD"]
		averageRepALLD[j] += AllDFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLD"]["ALLD"]

		averageRepDISC[j] += AllCFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLC"]["DISC"]
		averageRepDISC[j] += DiscFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["DISC"]["DISC"]
		averageRepDISC[j] += AllDFreq[j] * self.statisticsList[j]["reputations"]["viewsFromTo"]["ALLD"]["DISC"]



	for i, strat in enumerate(strats):
		plt.subplot(numPlotsCol, numPlotsRow, i + 1)
		plt.title("Reputations of Strategies as Viewed by " + strat)

		ALLCHist = [self.statisticsList[j]["reputations"]["viewsFromTo"][strat]["ALLC"] for j in range(NUMGENERATIONS)]
		ALLDHist = [self.statisticsList[j]["reputations"]["viewsFromTo"][strat]["ALLD"] for j in range(NUMGENERATIONS)]
		DISCHist = [self.statisticsList[j]["reputations"]["viewsFromTo"][strat]["DISC"] for j in range(NUMGENERATIONS)]

		plt.plot(ALLCHist, "g-")
		plt.plot(DISCHist, "y-")
		plt.plot(ALLDHist, "r-")

		plt.plot(averageRepALLC, color = "lightgreen", linestyle = "dashed")
		plt.plot(averageRepDISC, color = "palegoldenrod", linestyle = "dashed")
		plt.plot(averageRepALLD, color = "lightcoral", linestyle = "dashed")





	plt.show()
	



# ------------------- Simulation Farming ---------------------

# with Pool(4) as p:
# 	statsList = p.map(evolve, range(NUMPOPULATIONS))

# print("Simulated " + str(len(statsList)) + " populations!")

#-------------------- Simulation Farm Statistics Analysis -----------------

# statsList[0].plotComprehensive()

#------------------- Individual Simulation -------------------

stats = evolve(0)
stats.plotComprehensive()




