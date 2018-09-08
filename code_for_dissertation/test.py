import sys
import os
decaynum = int(sys.argv[1])
figurepath="figures_linearode/decay-"+str(decaynum)

if not os.path.isdir(figurepath):
    os.makedirs(figurepath)