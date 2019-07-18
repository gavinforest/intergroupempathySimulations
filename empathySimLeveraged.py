# ---------------- Imports and top level parameters --------------------
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools as itr
from multiprocessing import Pool
import time
import sys
import argparse
import julia
import simulationPlotting as simPlot
import pickle
import os

J = julia.Julia()
juliaSim = J.include('populationSimulationController.jl')
print("PyJulia has nprocs: " + str(J.nprocs()))


# -------------------------  Flags, Consants, Parsers, Utilties ----------------------------

from constantsAndParsers import *
from utilities import *


# --------------- Statistics and Plotting ---------------------

class populationStatistics:
	def __init__(self, numGenerations, argTuple):
		self.statisticsList = [ None for k in range(numGenerations)]
		self.numGenerations = numGenerations
		self.popParams = arrayToPop(argTuple[0])
		self.envParams = arrayToEnv(argTuple[1])
		self.norm = (argTuple[2], argTuple[3])
		self.empathy = argTuple[4]
		self.wholeArgTuple = argTuple

	def plotTypes(self):
		type0popSequence = [stat["proportions"]["type0"]["total"] for stat in statistics.statisticsList]
		type1popSequence = [stat["proportions"]["type1"]["total"] for stat in statistics.statisticsList]

		plt.plot(type0popSequence, 'bo')
		plt.plot(type1popSequence, "g-")
		plt.show()


def plotComprehensive(stats, headerString = ""):

	plotDict = createPlotable(stats)
	strats = ["ALLD", "DISC", "ALLC"]

	# ----------- Formatting ----------

	lineFormats = {}
	for thing in ["color", "linestyle"]:
		lineFormats[thing] = {}
		for strat in strats + ["total", "intergroup", "intragroup"]:
			lineFormats[thing][strat] = {}

	lineFormats["color"]["ALLC"]["direct"] = "green"
	lineFormats["color"]["ALLC"]["average"] = "lightgreen"
	lineFormats["color"]["DISC"]["direct"] = "yellow"
	lineFormats["color"]["DISC"]["average"] = "palegoldenrod"
	lineFormats["color"]["ALLD"]["direct"] = "red"
	lineFormats["color"]["ALLD"]["average"] = "lightcoral"
	lineFormats["color"]["total"]["average"] = "grey"
	lineFormats["color"]["total"]["direct"] = "grey"
	lineFormats["color"]["intergroup"]["average"] = "violet"
	lineFormats["color"]["intergroup"]["direct"] = "darkviolet"
	lineFormats["color"]["intragroup"]["average"] = "lightpink"
	lineFormats["color"]["intragroup"]["direct"] = "pink"


	for thing in strats + ["total", "intergroup", "intragroup"]:
		lineFormats["linestyle"][thing]["average"] = "dashed"
		lineFormats["linestyle"][thing]["direct"] = "solid"

	lineFormats["linestyle"]["total"]["direct"] = "dashed"

	# ------------- Plotting -------------
	fig = plt.figure(1)
	numPlotsCol = 3
	numPlotsRow = 1
	if len(headerString) > 0:
		fig.suptitle(headerString)

	plotInd = 1

	plt.subplot(numPlotsCol,numPlotsRow,plotInd)
	plt.title(headerString + "Cooperation Rate by Strategy")
	plt.xlabel("Generations")
	plt.ylabel("Cooperation Rate")

	for thing in strats + ["total", "intergroup", "intragroup"]:

		plt.plot(plotDict["coops"][thing], color=lineFormats["color"][thing]["direct"], 
											linestyle = lineFormats["linestyle"][thing]["direct"],
											label = thing)
	plt.legend()
	plotInd += 1



	plt.subplot(numPlotsCol,numPlotsRow, plotInd)
	plt.title("Type A Strategy Frequencies")
	plt.xlabel("Generations")
	plt.ylabel("Frequency")

	for strat in strats:
		plt.plot(plotDict["freqs"]["type0"][strat], color = lineFormats["color"][strat]["direct"], 
													linestyle= lineFormats["linestyle"][strat]["direct"])
		plt.plot(plotDict["freqs"][strat + "s"], color = lineFormats["color"][strat]["average"],
												linestyle= lineFormats["linestyle"][strat]["average"])

	plotInd += 1

	plt.subplot(numPlotsCol,numPlotsRow, plotInd)
	plt.title("Type B Strategy Frequencies")
	plt.xlabel("Generations")
	plt.ylabel("Frequency")

	for strat in strats:
		plt.plot(plotDict["freqs"]["type1"][strat], color = lineFormats["color"][strat]["direct"], 
													linestyle= lineFormats["linestyle"][strat]["direct"])
		plt.plot(plotDict["freqs"][strat + "s"], color = lineFormats["color"][strat]["average"],
												linestyle= lineFormats["linestyle"][strat]["average"])

	fig = plt.figure(2)
	if len(headerString) > 0:
		fig.suptitle(headerString)


	for i, strat in enumerate(strats):
		plt.subplot(numPlotsCol, numPlotsRow, i + 1)
		plt.title("Reputations of Strategies as Viewed by " + strat)

		for toStrat in strats:
			plt.plot(plotDict["reputationViewsFromTo"][strat][toStrat], color = lineFormats["color"][toStrat]["direct"],
																		linestyle = lineFormats["linestyle"][toStrat]["direct"])

			plt.plot(plotDict["averageReps"][toStrat], color = lineFormats["color"][toStrat]["average"],
														linestyle = lineFormats["linestyle"][toStrat]["average"])


	fig = plt.figure(3)
	if len(headerString) > 0:
		fig.suptitle(headerString)

	plt.subplot(1,1,1)
	plt.title("Average Round Payoffs by Strategy")

	for strat in strats:
		plt.plot(plotDict["averagePayoffs"][strat], color = lineFormats["color"][strat]["direct"], label = strat)

	plt.legend()
	plt.show()


#-------------------- Multiple Simulation Analysis Functions -----------------

def plotMultiplePopulations(statLists):
	plotDicts = [createPlotable(stat) for stat in statLists]
	plotDict = deepAverage(plotDicts)
	plotComprehensive(plotDicts, "Averaged Over " + str(len(statLists)) + " Runs")
	

# ------------------- Simulation Farming Utilities ---------------------

def juliaOutputToStatisticsObject(output, argTuple):
	statObject = populationStatistics(len(output), argTuple)
	statObject.statisticsList = [None for i in range(len(output))]

	for i, generation in enumerate(output):
		coopM = generation['cooperationRate']
		repM = generation['reputationsViewFromTo']
		propM = generation['proportions']
		payM = generation['roundPayoffs']

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

		cooperationDict = {"total": coopM[3,0], "ALLC": coopM[2,0], "DISC": coopM[1,0], "ALLD": coopM[0,0], 
							"intragroup0":coopM[4,0],	"intergroup0->1" : coopM[5,0], "intergroup1->0" : coopM[6,0],
							"intragroup1" : coopM[7,0], "intergroup" : coopM[8,0]}

		stat = {}
		stat["proportions"] = proportionDict
		stat["cooperationRate"] = cooperationDict
		stat["reputations"] = reputationDict
		stat["roundPayoffs"] = {"ALLD": payM[0], "DISC":payM[1], "ALLC":payM[2]}

		statObject.statisticsList[i] = stat

	return statObject


def generateParameterTuples(parameterVariabilitySets, repeats):
	#parameter variability sets are dictionaries of the form
	#{ paramater: [values it can take]}
	#we then try all possible combinations of each

	individualSettings = [[] for key in parameterVariabilitySets.keys()]
	for i, key in enumerate(parameterVariabilitySets.keys()):
		individualSettings[i] = [(key, item) for item in parameterVariabilitySets[key]]

	individualSettings = tuple(individualSettings)
	# print(individualSettings)
	# print(*individualSettings)
	combos = list(itr.product(*individualSettings))
	# print("gen combos are: " + str(list(combos)))

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

				if type(item) != tuple:
					norm = (item, item)
				else:
					norm = item

			elif key == "empathy":
				empathy = item

			else:
				print("*bad key in generateParameterTuples: " + str(key))

		for j in range(repeats):
			finalTuples.append((popToArray(popParams),envToArray(envParams), norm[0], norm[1], empathy))
	combos = list(itr.chain.from_iterable([[combo] * repeats for combo in combos]))
	print("Length of combos list is " + str(len(combos)))
	print("Length of final tuples list is " + str(len(finalTuples)))

	return finalTuples, combos

def farm(parameterVariabilitySets, printing, caching = True, repeats = 1):
	# if type(parameterVariabilitySets) is dict:
	argumentTuples, variedValues = generateParameterTuples(parameterVariabilitySets, repeats)
	# else:


	statsList = J.evolveDistributed(argumentTuples,printing)
	statsObjs = []
	for obj, argTuple in zip(statsList, argumentTuples): 
		statsObjs.append(juliaOutputToStatisticsObject(obj,argTuple))
		if printing:
			print("converted an output to statistics object")

		if caching:
			fnameBase = "".join(str(time.time()).split("."))
			fnameStat = "pickles/" + fnameBase + "statobject.pkl"
			fnameArgTuple = "pickles/" + fnameBase + "argTuple.pkl"
			with open(fnameStat, "wb") as f:
				pickle.dump(statsObjs[-1], f)
			with open(fnameArgTuple, "wb") as f:
				pickle.dump(argTuple, f)





	if DEBUG:
		print("in farm --- statsObjs: " + str(statsObjs))
		print("in farm --- argumentTuples: " + str(argumentTuples))
		print("in farm --- variedValues: " + str(list(variedValues)))

	return list(zip(statsObjs, argumentTuples, variedValues))


def comboToDict(combo):
	ret = {}
	for key,item in combo:
		ret[key] = item
	return ret

# ---------------------- High Level Direction ---------------------



def makeFig3():
	if not TEST:
		repeats = 5
		paramVariabilitySets = {"norm": [SIMPLESTANDING, STERNJUDGING, SCORING, SHUNNING],
								"empathy": defaultEmpathyLevels, "ustrat": [0.0005, 0.0025, 0.01], "numGenerations":[20000]}
	else:
		repeats = 1
		# empathyLevels = [np.ones((2,2), dtype="float64") * (i / 2.0) for i in range(3)]
		paramVariabilitySets = {"norm": [SIMPLESTANDING, STERNJUDGING, SCORING, SHUNNING][0:1],
								"empathy": defaultEmpathyLevels, "ustrat": [0.0005], "numGenerations":[150000]}
	
	stats = farm(paramVariabilitySets, PROGRESSVERBOSE, repeats)

	if DEBUG:
		print("stats: " + str(list(stats)))

	fig = plt.figure(1)
	fig.set_size_inches((5 * len(paramVariabilitySets["ustrat"]),5))
	# plt.subplot(1,3,1)
	plotDict = {}
	for i in range(int(len(stats) / repeats)):
		statObj, paramTuple, combo = stats[repeats * i]
		# print("In stat loop")
		d = comboToDict(combo)
		ustrat = d["ustrat"]

		if ustrat not in plotDict:
			plotDict[ustrat] = {}

		normName = normToAbbreviation(d["norm"][0])
		if normName not in plotDict[ustrat]:
			plotDict[ustrat][normName] = {}
			plotDict[ustrat][normName]["empathy"] = []
			plotDict[ustrat][normName]["cooperation"] = []


		plotDict[ustrat][normName]["empathy"].append(np.average(d["empathy"]))


		averageCooperation = np.average([[stat["cooperationRate"]["total"] for stat in stats[repeats * i + j][0].statisticsList] 
										for j in range(repeats)])

		plotDict[ustrat][normName]["cooperation"].append(averageCooperation)

	if DEBUG:
		print("plotDict: " + str(plotDict))


	lineColorsDict = {"SJ": "blue", "SS": "green", "SC": "red", "SH": "purple"}
	numPlots = len(plotDict.keys())
	for i, ustratLevel in enumerate(sorted(plotDict.keys())):
		if PROGRESSVERBOSE:
			print("plotting u = " + str(ustratLevel))


		plt.subplot(1,numPlots,i+1)
		plt.title("u = " + str(ustratLevel))
		for normName in plotDict[ustratLevel].keys():
			plt.plot("empathy", "cooperation", data=plotDict[ustratLevel][normName], label=normName, 
				markersize=4, color=lineColorsDict[normName])

		axes = plt.gca()
		axes.set_xlim([0.0,1.0])
		axes.set_ylim([0.0,1.0])

		plt.legend()
		plt.xlabel("Empathy")
		plt.ylabel("Cooperation Rate")

	save = False
	if len(sys.argv) > 2:
		if sys.argv[2] == "file":
			save = True


	if not save:
		plt.show()

	else:
		fig.savefig("figures/" + str(time.time()).split('.')[0] + "empathyFigure3.pdf", bbox_inches='tight')


	return

# def investigateTimeScale(paramVariabilitySets):

def singleParameterRun():

	paramChanges = paramVariabilitySets = {"norm": SIMPLESTANDING,
								"empathy": EMPATHYTEMPLATES["unilateral01"], "ustrat": 0.0005, "numGenerations":50000,
								"intergroupUpdateP" : 0.0}

	for key in paramVariabilitySets.keys():
		paramChanges[key] = [paramChanges[key]]

	argTuples, _ = generateParameterTuples(paramChanges,1)

	tup = argTuples[0]


	stats = juliaOutputToStatisticsObject(J.singleRun(tup, PROGRESSVERBOSE), tup)
	plotComprehensive(stats)



def runFromJson(filename, printing, save, caching):
	with open(filename) as f:
		paramObject = json.load(f)

	if printing:
		print("Json loaded object: " + str(paramObject))

	# paramObject["variabilitySets"] = [jsonVariabilitySetParser(obj) in paramObject["variabilitySets"]]
	paramObject["variabilitySets"] = jsonVariabilitySetParser(paramObject["variabilitySets"])
	variabilitySets = paramObject["variabilitySets"]

	repeats = 1
	if "repeats" in paramObject:
		repeats = paramObject["repeats"]


	zipped = farm(variabilitySets, printing, caching, repeats)


	simPlot.generalPlotter(paramObject, zipped, printing, save)




#-------------------- Command Line Parsing and Control -----------------

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type = str, help="parameter input json file")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("function", help="", choices=["fig3", "singlerun", "json"])
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-c", "--cache", action="store_true")
args = parser.parse_args()

if len(sys.argv) > 1:
	if args.function == "fig3":
		makeFig3()
	elif args.function == "singlerun":
		singleParameterRun()
	else:
		runFromJson(args.input, args.verbose, args.save, args.cache)








