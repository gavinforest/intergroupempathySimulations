import matplotlib.pyplot as plt
import numpy as np

from constantsAndParsers import *
from dataextractors import *
from utilities import *







def searchstats(plotParam, Stats):
	#plotParam assumed to be dictionary
	stats = [s[0] for s in Stats[:]]
	for name in plotParam.keys():
		if name == "norm":
			stats = [stat for stat in stats if normToAbbreviation(stats.norm) == plotParam["norm"]]
		elif name == "empathy":
			empathyMatcher = lambda x: np.array_equal(x, plotParam["empathy"])
			stats = [stat for stat in stats if empathyMatcher(stat.empathy)]
		elif name in defaultEnvNames:
			stats = [stat for stat in stats if stat.envParams[name] == plotParam[name]]
		elif name in defaultPopNames:
			stats = [stat for stat in stats if stat.popParams[name] == plotParam[name]]
		elif name == "plotName":
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


	xAxisExtractors = {param["plotName"] : extractors[param["xextractor"]] for param in plottingParameters["extractors"] if "xextractor" in param}

	#default y axis extractor is cooperation rate
	yAxisExtractors = {param["plotName"] : extractors[param["yextractor"]] for param in plottingParameters["extractors"] if "yextractor" in param}


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
		yextractor = None

		plotName = plotParam["plotName"]

		if plotName in xAxisExtractors:
			xextractor = xAxisExtractors[plotName]
		else:
			xextractor = defaultXExtractor

		if plotName in yAxisExtractors:
			yextractor = yAxisExtractors[plotName]
		else:
			yextractor = defaultYExtractor



		plt.subplot(numRows, int(width / 5), i + 1)
		plt.title(plotName)
		plt.xlabel(xextractor.name)
		plt.ylabel(yextractor.name)

		series = []

		if "generational" not in yextractor.type and xextractor.name == "generations":
			raise ValueError("Bad JSON, paired yextractor " + yextractor.name + " with generations xextractor -- misaligned types")

		if "generational" in yextractor.type and xextractor.name == "generations":
			for stat in specificStats:
				series.append({"y": yextractor(stat), "x": xextractor(stat)})

		else:
			for stat in specificStats:
				series.append((xextractor(stat), yextractor(stat)))


		if printing:
			# print("series: " + str(series))
			pass


		if type(series[0]) == tuple:
			plt.scatter([x for x,y in series], [y for x,y in series])
			things = {}
			for x,y in series:

				if x not in things:
					things[x] = []

				things[x].append(y)

			for key in things.keys():
				things[key] = np.average(things[key])

			pairs = [(key, things[key]) for key in things.keys()]
			plt.plot([x for x,y in pairs], [y for x,y in pairs])

		else:
			for d in series:
				plt.plot("x", "y", 'o', data = d)

			averaged = deepAverage(series)
			plt.plot("x", "y", data = averaged, linestyle="solid")

	if savefig:
		fig.savefig("figures/" + str(time.time()).split('.')[0] + "empathyrun.pdf", bbox_inches='tight')

	else:
		plt.show()




