#!/bin/bash

source setup.sh
source ticlenv/bin/activate

# run the desired code
N=500
echo ${N}
for nEdg in 1 2 3
do
	python3 energyProfileLayerCutRun.py -o dataLayerCut50vs25_EProf_N${N}_nEdg${nEdg}.pkl --Ntr=${N} -nE=${nEdg}
done

xrdcp -f *.pkl root://eosuser.cern.ch//eos/user/d/dmagdali/pickle
rm *.pkl
