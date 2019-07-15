import numpy as np

def deepAverage(dictionaries): # I am so proud of this function
	for key in dictionaries[0].keys():
		if type(dictionaries[0][key]) == dict:
			dictionaries[0][key] = deepAverage([d[key] for d in dictionaries])
		else:
			dictionaries[0][key] = [np.average(el) for el in zip(*tuple([d[key] for d in dictionaries]))]

	return dictionaries[0]