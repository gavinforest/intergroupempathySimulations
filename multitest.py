from multiprocessing import Pool
import julia
J= julia.Julia()
J.include("multitest.jl")
with Pool(4) as p:
    print(p.map(J.f,range(5)))