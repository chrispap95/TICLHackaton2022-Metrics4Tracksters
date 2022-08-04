#!/bin/bash
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node

# setting up the input variables
#dataset=$1

#inputDir=root://cmseos.fnal.gov//store/user/chpapage/SUEP_svjprod
#nameIn=step_GEN_mMed${mMed}_mDark${mDark}_temp${temp}_gen13TeV_${scenario}
#nameOut=${nameIn}_PT${PT}_iso${geometry}${N}

# bring in the tarball you created before with caches and large files excluded:
xrdcp -s root://eosuser.cern.ch//eos/user/d/dmagdali/TICLHackaton2022-Metrics4Tracksters.tgz .
tar -xf TICLHackaton2022-Metrics4Tracksters.tgz
rm TICLHackaton2022-Metrics4Tracksters.tgz
cd TICLHackaton2022-Metrics4Tracksters
source setup.sh
source ticlenv/bin/activate

# run the desired code
#"c_nxkatz" "c_nxeigen" "c_nxpr"
N=500
echo ${N}
for nEdg in 1
do
	for cen in "c_pr" 
	do
		for isDir in 0 1
		do
			python3 centralityProfileLayerCutRun.py -f test.txt -o dataLayerCut50vs25_N${N}_${cen}_nEdg${nEdg}_Dir${isDir}.pkl --Ntr=${N} -nE=${nEdg} -Dir=${isDir} -c ${cen}
		done
	done
done

xrdcp -f *.pkl root://eosuser.cern.ch//eos/user/d/dmagdali/pickle
rm *.pkl
