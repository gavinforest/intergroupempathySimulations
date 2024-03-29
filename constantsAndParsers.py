import numpy as np

#----------------------- Global Constants ----------------------



DEBUG = False
PROGRESSVERBOSE = True
TEST = True

defaultPopNames = ["numAgents", "numGenerations"]
defaultEnvNames = ["Ecoop", "Eobs", "ustrat", "u01", "u10", "w", "gameBenefit", "gameCost", "intergroupUpdateP"]

def popToArray(x):
	keys = defaultPopNames
	return np.array([x[key] for key in keys])

def envToArray(x):
	keys = defaultEnvNames
	return np.array([x[key] for key in keys])

def arrayToPop(x):
	keys = defaultPopNames
	return {key:x[i] for i,key in enumerate(keys)}

def arrayToEnv(x):
	keys = defaultEnvNames
	return {key:x[i] for i,key in enumerate(keys)}

SIMPLESTANDING = np.array([[1, 0], [1, 1]], dtype="int64")
STERNJUDGING = np.array([[1, 0], [0, 1]], dtype="int64")
SCORING = np.array([[0, 0], [1, 1]], dtype="int64")
SHUNNING = np.array([[0, 0], [0, 1]], dtype="int64")

normAbbreviations = [(SIMPLESTANDING, "SS"), (STERNJUDGING, "SJ"), (SCORING, "SC"), (SHUNNING, "SH")]
abbreviationToNorm = {"SS" : SIMPLESTANDING, "SJ":STERNJUDGING, "SC":SCORING, "SH":SHUNNING}


def normToAbbreviation(norm, namer = lambda x: "Unknown Name Norm"):
	for n, abbrv in normAbbreviations:
		if np.array_equal(norm, n):
			return abbrv

	return namer(norm)



defaultPopulationParameters = {"numAgents" : 100, "numGenerations" : 5000}
defaultEnvironmentParameters = {"Ecoop" : 0.02, "Eobs" : 0.02, "ustrat" : 0.001, "u01" : 0.0, "u10" : 0.0, "w" : 1.0,
					"gameBenefit" : 5.0, "gameCost" : 1.0, "intergroupUpdateP" : 0.5}
defaultNorm = (np.copy(SIMPLESTANDING), np.copy(SIMPLESTANDING))
defaultEmpathy = np.zeros((2,2), dtype="float64")

defaultEmpathyLevels = [np.ones((2,2), dtype="float64") * (i / 5.0) for i in range(6)]




# ---------------- Various Parsers ----------------


def createPlotable(statObject):

	strats = ["ALLD", "DISC", "ALLC"]

	plotAbleDict = {"coops" : {}, 
					"freqs" : {"type1" : {"ALLC" : [],"DISC" : [], "ALLD" : []}, "type0" : {"ALLC" : [],"DISC" : [], "ALLD" : []}}}

	for entry in statObject.statisticsList:
		for typ in ["type0", "type1"]:
			for strat in ["ALLC","DISC", "ALLD"]:
				plotAbleDict["freqs"][typ][strat].append(1.0 * entry["proportions"][typ][strat] / entry["proportions"][typ]["total"])

	allCoops = [entry["cooperationRate"]  for entry in statObject.statisticsList]
	for name in strats + ["total", "intergroup", "intragroup"]:
		plotAbleDict["coops"][name] = [entry[name] for entry in allCoops]




	for strat in ["ALLC","DISC", "ALLD"]:
		plotAbleDict["freqs"][strat + "s"] = [(a + b) / 2.0 for a,b in 
											zip(plotAbleDict["freqs"]["type0"][strat], plotAbleDict["freqs"]["type1"][strat])]
	

	plotAbleDict["averageReps"] = {}
	for strat in strats:
		plotAbleDict["averageReps"][strat] = [0 for j in range(statObject.numGenerations)]

	for j in range(statObject.numGenerations):
		for fromStrat in strats:
			for toStrat in strats:
				plotAbleDict["averageReps"][toStrat][j] += plotAbleDict["freqs"][fromStrat + "s"][j] * statObject.statisticsList[j]["reputations"]["viewsFromTo"][fromStrat][toStrat]

	plotAbleDict["reputationViewsFromTo"] = {}
	for fs in strats:
		plotAbleDict["reputationViewsFromTo"][fs] = {}
		for ts in strats:
			plotAbleDict["reputationViewsFromTo"][fs][ts] = [statObject.statisticsList[j]["reputations"]["viewsFromTo"][fs][ts] 
																for j in range(statObject.numGenerations)]

	avgPayoffs = {}
	for strat in strats:
		avgPayoffs[strat] = [statObject.statisticsList[j]["roundPayoffs"][strat] for j in range(statObject.numGenerations)]

	plotAbleDict["averagePayoffs"] = avgPayoffs

	return plotAbleDict



# ----------------------------- Constants for naming complicated things in parameter Json ---------------------------
EMPATHYTEMPLATES = {"unilateral01" : np.array([[1,1],[0,0]], dtype = "float64"), "egalitarian" : np.ones((2,2), dtype="float64"),
					"unilateral10" : np.array([[0,0],[1,1]], dtype = "float64"), "shunning" : np.array([[1,0],[0,1]], dtype = "float64")}

def genEmpathies(command):
	commandList = command.split(" ")
	number = int(commandList[1])
	typ = commandList[2]

	maximumE = 1.0
	if len(commandList) > 3:
		setting = commandList[3]
		if setting == "max":
			maximumE =  float(commandList[4])


	empathyList = []

	if typ == "unilateral01GrowEven":
		base = EMPATHYTEMPLATES["unilateral01"]

		added = EMPATHYTEMPLATES["unilateral10"]

	elif typ == "intergroupGrow":
		base = EMPATHYTEMPLATES["shunning"]

		added = np.array([[0,1],[1,0]], dtype="float64")

	
	for i in range(number):
			empathyList.append(base + added * maximumE * (i / (number - 1.0)))

	return empathyList





def jsonVariabilitySetParser(dicty):
	if "norm" in dicty:
		newNorms = []
		for name in dicty["norm"]:
			if type(name) == str:
				norm = abbreviationToNorm[name]
				newNorms.append((norm, norm))
			else:
				newNorms.append((abbreviationToNorm[name[0]], abbreviationToNorm[name[1]]))

		dicty["norm"] = newNorms
	if "empathy" in dicty:
		empathyList = []
		for el in dicty["empathy"]:
			if type(el) == float:
				empathyList.append(np.ones((2,2), dtype="float64") * el)
			elif type(el) == str:
				empathyList = empathyList + genEmpathies(el)
			elif type(el) == list:
				empathyList.append(np.array(el), dtype="float64")

		dicty["empathy"] = empathyList

	return dicty

def parsedJsonToJsonable(params):
	parsedPlots = []
	for plot in params["plots"]:
		if "norm" in plot:
			plot["norm"] = tuple(map(normToAbbreviation, plot["norm"]))
		if "empathy" in plot:
			plot["empathy"] = np.average(plot["empathy"])
		parsedPlots.append(plot)


	params["plots"] = parsedPlots
	if "empathy" in params["variabilitySets"]:
		params["variabilitySets"]["empathy"] = [thing.tolist() for thing in params["variabilitySets"]["empathy"]]

	if "norm" in params["variabilitySets"]:
		params["variabilitySets"]["norm"] = [list(map( normToAbbreviation, thing)) for thing in params["variabilitySets"]["norm"]]





