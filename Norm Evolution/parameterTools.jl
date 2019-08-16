module parameterTools

export modifyParameters, validateRunParameters

const constantParams = Set(["NUMGROUPS", "NUMAGENTSPERNORM", "NUMGENERATIONS","BATCHSPERAGENT","BATCHSIZE"])

function validateSingleParameter(runParam)
	for j in keys(runParam)
		if in(j, constantParams)
			return false
		end
	end
	return true
end

function validateRunParameters(runParameters)
	for i in keys(runParameters)
		if ! validateSingleParameter(runParameters[i]) && (i != "startState")
			return false
		end
	end
	return true
end

function modifyParameters(simParameters, state, runParametersMods)
	if ! validateSingleParameter(runParametersMods)
		return "failed"
	end

	for k in keys(runParametersMods)
		simParameters[k] = runParametersMods[k]
	end
	return "succeeded"

end



end
