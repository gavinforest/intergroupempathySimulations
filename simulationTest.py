import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

import julia
J= julia.Julia()
# J.include("imageMatrixTesting.jl")
J.include("populationSimulationSimple.jl")
# J.include("populationSimulation.jl")

# imageMatrices = J.interactionMatrix()
# imageMatrices = J.semiagentInteractionMatrix()
# imageMatrix = J.interactionIterateCondensed()
stats, imageMatrix = J.evolve()

# print(len(imageMatrices))

fig = plt.figure()

ax1 = plt.subplot(1,1,1)

# def animationFunc(i):
# 	ax1.matshow(imageMatrices[i % len(imageMatrices)])

# def semiAgentAvg(mat):
# 	avged = np.ones((16,16), dtype="Float64")

# 	for i in range(mat.shape[0]):
# 		for j in range(mat.shape[0]):
# 			avged[i % 16, j % 16] += mat[i,j]

# 	avged = avged / (mat.shape[0] / 16) **2
# 	return avged



# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# ani = animation.FuncAnimation(fig, animationFunc, interval=1000)
# ani.save("1616approxtest1.gif", writer="imagemagick")

# plt.matshow(imageMatrix, cmap="Greys_r")
# plt.colorbar()
for i in range(16):
	line = []
	for stat in stats:
		line.append(stat[0][i])

	plt.plot(line, label = "norm " + str(i+1))

coops = [stat[1] for stat in stats]
plt.plot(coops, label="Cooperation Rate", color="grey", linestyle="dashed")

plt.legend()
plt.show()