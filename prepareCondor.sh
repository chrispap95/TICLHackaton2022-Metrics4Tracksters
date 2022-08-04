#!/bin/sh
#USERBASE=`pwd`
#rm ${CMSSW_VERSION}.tgz
cd ..
echo "Creating tarball..."
tar --exclude=*.root --exclude=*.ipynb --exclude=figures --exclude=pickle --exclude=test --exclude-vcs -zcvf TICLHackaton2022-Metrics4Tracksters.tgz TICLHackaton2022-Metrics4Tracksters
#xrdcp -f TICLHackaton2022-Metrics4Tracksters.tgz root://afsuser.cern.ch//afs/cern.ch/user/d/dmagdali/CernCentralityProject/TICLHackaton2022-Metrics4Tracksters.tgz
if [ ! -f TICLHackaton2022-Metrics4Tracksters.tgz ]; then
echo "Error: tarball doesn't exist!"
else
echo " Done!"
fi
#rm TICLHackaton2022-Metrics4Tracksters.tgz
