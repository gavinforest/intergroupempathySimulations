import numpy as np
from constantsAndParsers import *

class generations(object):
	def __init__(self):
		self.name = "generations"

	def __call__(self, statObj):
		return list(range(statObj.numGenerations))

class cooperationRateTT(object):
	def __init__(self):
		self.name = "cooperation rate total total"
		self.type = "full"

	def __call__(self,statObj):
		return np.average([statDict["cooperationRate"]["total"] for statDict in statObj.statisticsList])


class cooperationRateTG(object):
	def __init__(self):
		self.name = "cooperation rate total generational"
		self.type = "generational"

	def __call__(self,statObj):
		statDict = createPlotable(statObj)
		return statDict["coops"]["total"]

class cooperationRateTInter(object):
	def __init__(self):
		self.name = "cooperation rate intergroup total"
		self.type = "full"

	def __call__(self,statObj):
		return np.average([statDict["cooperationRate"]["intergroup"] for statDict in statObj.statisticsList])


class cooperationRateTIntra(object):
	def __init__(self):
		self.name = "cooperation rate intragroup total"
		self.type = "full"

	def __call__(self,statObj):
		return np.average([statDict["cooperationRate"]["intragroup"] for statDict in statObj.statisticsList])

class simpleEmpathy(object):
	def __init__(self):
		self.name = "simple empathy"
		self.type = "full"

	def __call__(self,statObj):
		empMatrix = statObj.empathy
		return np.average(empMatrix)

class simpleEmpathyType1(object):
	def __init__(self):
		self.name = "simple empathy type1"
		self.type = "full"

	def __call__(self,statObj):
		empMatrix = statObj.empathy
		return np.average(empMatrix[1,:])

class simpleIntergroupEmpathy(object):
	def __init__(self):
		self.name = "simple intergroup empathy"
		self.type = "full"

	def __call__(self,statObj):
		empMatrix = statObj.empathy
		return (empMatrix[0,1] + empMatrix[1,0]) / 2.0

extractorList = [generations(), cooperationRateTT(), simpleEmpathy(), cooperationRateTG(), simpleEmpathyType1(), 
				simpleIntergroupEmpathy(), cooperationRateTInter(), cooperationRateTIntra()]


extractors = {obj.name : obj for obj in extractorList}