# ---------------- Imports and top level parameters --------------------
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools as itr
from multiprocessing import Pool
import time
import random
import ctypes
import julia
J = julia.Julia()
juliaSim = J.include('populationSimulationController.jl')


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

class populationStatistics:
	def __init__(self, numGenerations):
		self.statisticsList = [ None for k in range(numGenerations)]
		self.populationHistory = [None for k in range(numGenerations)]

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

def juliaOutputToStatisticsObject(output):
	statObject = populationStatistics(len(output))
	statObject.statisticsList = [None for i in range(len(output))]

	for i, generation in enumerate(output):
		coopM = generation['cooperationRate']
		repM = generation['reputationsViewFromTo']
		propM = generation['proportions']

		proportionDict = {}

		# TEST THIS
		proportionDict["type0"] = {"total": propM[0,3], "ALLD": propM[0,0], "DISC": propM[0,1], "ALLC" : propM[0,2] }
		proportionDict["type1"] = {"total": propM[1,3], "ALLD": propM[1,0], "DISC": propM[1,1], "ALLC" : propM[1,2] }


		reputationDict = {}

		#TEST THIS
		strats = ["ALLD", "DISC", "ALLC"]
		reputationDict["viewsFromTo"] = {}

		for strat in strats:
			reputationDict["viewsFromTo"][strat] = {}

		for j, vFrom in enumerate(strats):
			for k, tTo in enumerate(strats):
				reputationDict["viewsFromTo"][vFrom][tTo] = repM[j,k]

		cooperationDict = {"total": coopM[3,0], "ALLC": coopM[0,0], "DISC": coopM[1,0], "ALLD": coopM[2,0]}

		stat = {}
		stat["proportions"] = proportionDict
		stat["cooperationRate"] = cooperationDict
		stat["reputations"] = reputationDict

		statObject.statisticsList[i] = stat

	return statObject

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
defaultPopulationParameters = {"numAgents" : 100, "numGenerations" : 150000}
defaultEnvironmentParameters = {"Ecoop" : 0.0, "Eobs" : 0.0, "ustrat" : 0.001, "u01" : 0.0, "u10" : 0.0, "w" : 1.0,
					"gameBenefit" : 5.0, "gameCost" : 1.0}
defaultNorm = np.array([[1, 0], [0, 1]], dtype="int64")
defaultEmpathy = np.zeros((2,2), dtype="float64")


def popToArray(x):
	keys = ["numAgents", "numGenerations"]
	return np.array([x[key] for key in keys])

def envToArray(x):
	keys = ["Ecoop", "Eobs", "ustrat", "u01", "u10", "w", "gameBenefit", "gameCost"]
	return np.array([x[key] for key in keys])

def arrayToPop(x):
	keys = ["numAgents", "numGenerations"]
	return {key:x[i] for i,key in enumerate(keys)}

def arrayToEnv(x):
	keys = ["Ecoop", "Eobs", "ustrat", "u01", "u10", "w", "gameBenefit", "gameCost"]
	return {key:x[i] for i,key in enumerate(keys)}


def generateParameterTuples(parameterVariabilitySets):
	#parameter variability sets are dictionaries of the form
	#{ paramater: [values it can take]}
	#we then try all possible combinations of each

	individualSettings = [[] for key in parameterVariabilitySets.keys()]
	for i, key in enumerate(parameterVariabilitySets.keys()):
		individualSettings[i] = [(key, item) for item in parameterVariabilitySets[key]]

	individualSettings = tuple(individualSettings)
	# print(individualSettings)
	# print(*individualSettings)
	combos = itr.product(*individualSettings)
	# print(list(combos))

	finalTuples = []

	for setting in combos:
		popParams = defaultPopulationParameters.copy()
		envParams = defaultEnvironmentParameters.copy()
		norm = np.copy(defaultNorm)
		empathy = np.copy(defaultEmpathy)

		for key, item in setting:
			if key in popParams:
				popParams[key] = item
			elif key in envParams:
				envParams[key] = item
			elif key == "norm":
				norm = item
			elif key == "empathy":
				empathy = item

			else:
				print("*bad key in generateParameterTuples: " + str(key))

		finalTuples.append((popToArray(popParams),envToArray(envParams), norm, empathy))
	
	return finalTuples

def farm(parameterVariabilitySets):
	argumentTuples = generateParameterTuples(parameterVariabilitySets)
	statsList = J.evolveDistributed(argumentTuples)
	statsObjs = [juliaOutputToStatisticsObject(obj) for obj in statsList]

	return zip(statsObjs, argumentTuples)


def plotFarmedFig3(zippedStatsAndSets):
	plotLists = {}

	plotParamaters = []
	for statObj, tuples in zippedStatsAndSets:
		datapoint =      None
	return

def makeFig3():
	paramVariabilitySets = {}

# print(generateParameterTuples({"numGenerations": [100,200,300], "gameBenefit": [5,6,7]}))

paramSets = {"numGenerations": [10000], "gameBenefit": [5]}
x = list(farm(paramSets))
print(x)
x[0][0].plotComprehensive()
# print J.nprocs()

# with Pool(4) as p:
# 	statsList = p.map(evolve, range(NUMPOPULATIONS))

# print("Simulated " + str(len(statsList)) + " populations!")




#-------------------- Simulation Farm Statistics Analysis -----------------

# statsList[0].plotComprehensive()



#------------------- Individual Simulation -------------------

# stats = juliaOutputToStatisticsObject(J.testEvolve())
# print(stats.statisticsList[-1])

# stats.plotComprehensive()




