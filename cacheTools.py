import os
import pickle
import json
cachePath = "./cached"

def matchTarget(parameters, target, errormissing = true):
    for key in target:
        if key not in parameters:
            if errormissing:
                return False
            else:
                pass
        else:
            if target[key] != parameters[key]:
                return False
    return True
        

def loadObj(filename):
    if ".pkl" in filename:
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        with open(filename, "r") as f:
            return json.load(f)


def recoverRun(parameters):
    filenames = os.listdir(cachePath)
    objs = []
    for filename in filenames:
        obj = loadObj(filename)
            if matchTarget(obj["parameters"], parameters):
                objs.append(obj)
    return objs

def generateJsonCaches():
    filenames = os.listdir(cachePath)

    for filename in filenames:
        if ".json" not in filename:


