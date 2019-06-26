import numpy as np


NUMAGENTS = 100

NUMGENERATIONS = 1500

#observation error
Eobs = 0.0

#cooperation error
Ecoop = 0.0

class DonationGame:
	def __init__(self, benefit, cost):
		self.b = benefit
		self.c = cost
		self.payoffMatrix = np.array([[0, benefit],[-cost, benefit - cost]])

	def play(self, strat1, strat2):
		if strat1 == 1:
			if np.random.random() < Ecoop:
				strat1 == 0

		if strat2 == 1:
			if np.random.random() < Ecoop:
				strat2 = 0

		
		return self.payoffMatrix[strat1,strat2], self.payoffMatrix[strat2, strat1]

class Agent:
	def __init__(self, AgentType, empathyIn = 0.0, empathyOut = 0.0, strat = None):
		self.type = AgentType
		self.Ei = empathyIn
		self.Eo = empathyOut
		if self.strategy != None:
			self.strategy = strat
		else:
			self.strategy = np.array([1,1])

		self.reputations = [1 for i in range(NUMAGENTS)]


game = DonationGame(1.3, 1)


population = [Agent(int(np.floor(i / 50.0))) for i in range(NUMAGENTS)]

for i in range(NUMGENERATIONS):
	# reputationUpdates = [[None for i in NUMAGENTS] ]
	for j, agent in enumerate(population):
		for k, adversary in enumerate(population):
			if j != k:
				agentRep = adversary.reputations[j]
				adversaryRep = agent.reputations[k]

				agentAction = agent.strategy[adversaryRep]
				adversaryAction = adversary.strategy[agentRep]

			

				elif

				agentPayoff, adversaryPayoff = game.play(agent.strat, adversary.strat)





