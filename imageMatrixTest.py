import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

import julia
J= julia.Julia()
J.include("imageMatrixTesting.jl")

# imageMatrices = J.interactionMatrix()
# imageMatrices = J.semiagentInteractionMatrix()
imageMatrix = J.interactionIterateCondensed()
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

plt.matshow(imageMatrix, cmap="Greys_r")
plt.colorbar()
plt.show()