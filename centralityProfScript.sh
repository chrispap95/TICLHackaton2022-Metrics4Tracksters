#!/bin/bash

source setup.sh
source ticlenv/bin/activate

# run the desired code
#"c_nxkatz" "c_nxeigen" "c_nxpr"
N=500
echo ${N}
for nEdg in 1 2 3
do
	for cen in "c_pr" 
	do
		for isDir in 0
		do
			python3 centralityProfileLayerCutRun.py -f test.txt -o dataLayerCut100vs50_N${N}_${cen}_nEdg${nEdg}_Dir${isDir}.pkl --Ntr=${N} -nE=${nEdg} -Dir=${isDir} -c ${cen}
		done
	done
done

xrdcp -f *.pkl root://eosuser.cern.ch//eos/user/d/dmagdali/pickle
rm *.pkl
