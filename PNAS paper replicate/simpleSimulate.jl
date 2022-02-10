include("norms.jl")

#include check
# explainActionRule(L1b)

struct agent
	assessmentRule
	actionRule
	currentPayoff::Float64
	reputationViews::Array{Int64,1}
	agentID::Int64
end


