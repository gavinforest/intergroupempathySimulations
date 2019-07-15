#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 60
#SBATCH -p serial_requeue
#SBATCH --mem=20000


module-load julia/1.1.1-fasrc01
module-load python/3.6.3-fasrc02
pip3 install julia numpy matplotlib
julia "using Pkg; Pkg.add(\"PyCall\")"
python3 empathySimLeveraged.py json -s -i unilateralParameters.json 





