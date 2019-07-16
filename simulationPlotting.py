import matplotlib.pyplot as plt
import numpy as np
import json
import time

from constantsAndParsers import *
from dataextractors import *
from utilities import *







def searchstats(plotParam, Stats):
	#plotParam assumed to be dictionary
	stats = [s[0] for s in Stats]
	for name in plotParam.keys():
		if name == "norm":
			stats = [stat for stat in stats if normToAbbreviation(stat.norm) == plotParam["norm"]]
		elif name == "empathy":
			empathyMatcher = lambda x: np.array_equal(x, plotParam["empathy"])
			stats = [stat for stat in stats if empathyMatcher(stat.empathy)]
		elif name in defaultEnvNames:
			stats = [stat for stat in stats if stat.envParams[name] == plotParam[name]]
		elif name in defaultPopNames:
			stats = [stat for stat in stats if stat.popParams[name] == plotParam[name]]
		elif name == "plotName" or name == "extractors":
			pass
		else:
			raise ValueError("Bad JSON object, unrecognized keyword: " + str(name))

	return stats


def generalPlotter(plottingParameters, stats, printing = False, savefig = True):

	if not savefig:
		print("risky not using savefig...")


	#stats assumed to be list of statObjs
	subPlots = plottingParameters["plots"]
	defaultXExtractor = extractors[plottingParameters["defaults"]["xextractor"]]
	defaultYExtractor = extractors[plottingParameters["defaults"]["yextractor"]]


	xAxisExtractors = {subplot["plotName"] : subplot["extractors"]["xextractor"] for subplot in subPlots if "xextractor" in subplot["extractors"]}

	#default y axis extractor is cooperation rate
	yAxisExtractors = {subplot["plotName"] : subplot["extractors"]["yextractors"] for subplot in subPlots if "yextractors" in subplot["extractors"]}

	for key in xAxisExtractors.keys():
		xAxisExtractors[key] = [extractors[name] for name in xAxisExtractors[key]]

	for key in yAxisExtractors.keys():
		yAxisExtractors[key] = [extractors[name] for name in yAxisExtractors[key]]


	if printing:
		print("xAxisExtractors : " + str(xAxisExtractors))
		print("yAxisExtractors : " + str(yAxisExtractors))


	fig = plt.figure(1)
	numRows = int(np.ceil(len(subPlots) / 3))

	width = len(subPlots) * 5
	if width > 15:
		width = 15

	if printing:
		print("width is: " + str(width))
	fig.set_size_inches((width, 5 * numRows))

	for i, plotParam in enumerate(subPlots):
		specificStats = searchstats(plotParam, stats)

		xextractor = None
		yextractors = None

		plotName = plotParam["plotName"]

		if plotName in xAxisExtractors:
			xextractor = xAxisExtractors[plotName]
		else:
			xextractor = defaultXExtractor

		if plotName in yAxisExtractors:
			yextractors = yAxisExtractors[plotName]
		else:
			yextractors = [defaultYExtractor]



		plt.subplot(numRows, int(width / 5), i + 1)
		plt.title(plotName)
		plt.xlabel(xextractor.name)
		plt.ylabel(yextractors[0].name)

		series = [[] for i in range(len(yextractors))]

		for i,yextractor in enumerate(yextractors):

			if "generational" not in yextractor.type and xextractor.name == "generations":
				raise ValueError("Bad JSON, paired yextractor " + yextractor.name + " with generations xextractor -- misaligned types")

			if "generational" in yextractor.type and xextractor.name == "generations":
				for stat in specificStats:
					series[i].append({"y": yextractor(stat), "x": xextractor(stat)})

			else:
				for stat in specificStats:
					series[i].append((xextractor(stat), yextractor(stat)))


		if printing:
			# print("series: " + str(series))
			pass


		if type(series[0][0]) == tuple:
			plt.scatter([x for x,y in series[0]], [y for x,y in series[0]])

			for i, subseries in enumerate(series):
				things = {}
				for x,y in subseries:

					if x not in things:
						things[x] = []

					things[x].append(y)

				for key in things.keys():
					things[key] = np.average(things[key])

				pairs = [(key, things[key]) for key in things.keys()]
				plt.plot([x for x,y in pairs], [y for x,y in pairs], label="averaged " + yextractors[i].name)

			plt.legend()
			axes = plt.gca()
			axes.set_ylim([0.0,1.0])

		else:
			for d in series[0]:
				plt.plot("x", "y", 'o', data = d, label = yextractors[0].name)

			for i, subseries in series:
				averaged = deepAverage(subseries)
				plt.plot("x", "y", data = averaged, linestyle="solid", label = "averaged " + yextractors[i].name)

			plt.legend()
			axes = plt.gca()
			axes.set_ylim([0.0,1.0])

	if savefig:
		timestamp = str(time.time()).split('.')[0]
		fig.savefig("figures/" + timestamp + "empathyrun.pdf", bbox_inches='tight')
		with open("figures/" + timestamp + "empathyrunSETTINGS.json", "w") as f:
			json.dump(parsedJsonToJsonable(plottingParameters), f)

	else:
		plt.show()




