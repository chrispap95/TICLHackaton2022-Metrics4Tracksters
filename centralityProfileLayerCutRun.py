import sys
#sys.path.insert(1, '../')
import src.Network as net
import awkward as ak
import src.functions as fn
import os
import pickle
import argparse
import uproot

parser = argparse.ArgumentParser(description="Centrality profile")
parser.add_argument("-f", "--file", dest="filename",
                  help="full path to file to process")
parser.add_argument("-o", "--output", dest="output",
                  help="local path for output file")
parser.add_argument("-N", "--Ntr",
                  action="store", dest="Ntr", type=int, default=100,
                  help="number of tracksters to process")
parser.add_argument("-nE", "--nEdg",
                  action="store", dest="nEdg", type=int, default=1,
                  help="number of edges")
parser.add_argument("-c", "--centrality",
                  action="store", dest="centrality", type=str, default="c_pr",
                  help="centrality used in method")
parser.add_argument("-Dir", "--isDir",
                  action="store", dest="isDirected", type=bool, default=False,
                  help="directed graph boolean")
options = parser.parse_args()

#print(type(options.Ntr))
#print(options.Ntr)
N=int(options.Ntr)
#print(N)
nEdg=options.nEdg
optCen=options.centrality
isDirected=options.isDirected

filenameFull100GeV="/eos/user/c/chpapage/TICL_samples/CloseByDoubleGamma_E100Eta1p62Delta5_CMSSW_12_4_0_upgrade2026_D86_clue3Dv4_ntuples/220701_225928/0000/ntuples.root"
filenameFull50GeV="root://eosuser.cern.ch//eos/user/c/chpapage/TICL_samples/CloseByDoubleGamma_E50Eta1p62Delta5_CMSSW_12_4_0_upgrade2026_D86_clue3Dv4_ntuples/220701_225808/0000/ntuples.root"
#filenameFull25GeV="root://eosuser.cern.ch//eos/user/c/chpapage/TICL_samples/CloseByDoubleGamma_E25Eta1p62Delta5_CMSSW_12_4_0_upgrade2026_D86_clue3Dv4_ntuples/220701_225704/0000/ntuples.root"
#folder="CloseByDoubleGamma_E50IncVsE25Com_Eta1p62Delta5_CMSSW_12_4_0_upgrade2026_D86_clue3Dv4"
#savefigs=False
#file = uproot.open(filename)
#fileFull=uproot.concatenate(filename)
fileCom=uproot.open(filenameFull50GeV)
fileInc=uproot.open(filenameFull100GeV)
#datasetName="50 GeV data inc(layerCut) vs 25 GeV data com"

tracksters=fileCom["ana/tracksters"]
vertices_E = tracksters['vertices_energy'].array()
vertices_indexes = tracksters['vertices_indexes'].array()
vertices_x = tracksters['vertices_x'].array()
vertices_y = tracksters['vertices_y'].array()
vertices_z = tracksters['vertices_z'].array()
vertices_layers=tracksters['vertices_layer'].array()

tracksters_inc=fileInc["ana/tracksters"]
vertices_E_inc = tracksters_inc['vertices_energy'].array()
vertices_indexes_inc = tracksters_inc['vertices_indexes'].array()
vertices_x_inc = tracksters_inc['vertices_x'].array()
vertices_y_inc = tracksters_inc['vertices_y'].array()
vertices_z_inc = tracksters_inc['vertices_z'].array()
vertices_layers_inc=tracksters_inc['vertices_layer'].array()


comCenProfArray=[]
comNVerticesList=[]
for evt in range(N):
	print(evt)
	for tr in range(min(len(vertices_indexes[evt]),2)):
		if(evt==121 and tr==1):
			continue
		#v_layers=vertices_layers[evt][tr]
		v_ind=vertices_indexes[evt][tr]
		#v_x=vertices_x[evt][tr]
		#v_y=vertices_y[evt][tr]
		#v_z=vertices_z[evt][tr]
		#v_E=vertices_E[evt][tr]
		TrNet=net.Network(v_ind,vertices_x[evt,tr],vertices_y[evt,tr],
								 vertices_z[evt,tr],vertices_E[evt,tr])
		edges_1 = TrNet.edgeBuilderNew(nEdg)
		edges_1 = ak.flatten(edges_1[ak.num(edges_1) > 0].to_list())
		if(optCen=="c_nxkatz"):
			centrality=TrNet.nXCentralityKatz(v_ind,edges_1,isDirected)
		elif(optCen=="c_nxpr"):
			centrality=TrNet.centralityPageRank(v_ind,edges_1,0.85,isDirected)
		elif(optCen=="c_pr"):
			centrality=TrNet.nXCentralityPageRank(v_ind,edges_1,0.85,isDirected)
		elif(optCen=="c_nxeigen"):
			if(len(v_ind)<3):
				continue
			centrality=TrNet.nXCentralityEigen(v_ind,edges_1,isDirected)
			
		adjMatrix=TrNet.adjM(v_ind,edges_1)

		cenProfList=TrNet.centralityProf(adjMatrix,centrality)
		comCenProfArray.append(cenProfList)
		comNVerticesList.append(len(v_ind))

with open(options.output,"wb") as f:
	pickle.dump(comCenProfArray, f)
	pickle.dump(comNVerticesList, f)

del comCenProfArray
del comNVerticesList

incCenProfArray=[]
incNVerticesList=[]
#incEnergies=[]
for evt in range(N):
	print(evt)
	for tr in range(min(len(vertices_indexes_inc[evt]),2)):
		if(evt==121 and tr==1):
			continue
		incSlice=fn.incompleteTracksters(vertices_layers_inc[evt,tr],0.31,0.05)
		v_ind_inc=vertices_indexes_inc[evt,tr][incSlice]
		v_x_inc=vertices_x_inc[evt,tr][incSlice]
		v_y_inc=vertices_y_inc[evt,tr][incSlice]
		v_z_inc=vertices_z_inc[evt,tr][incSlice]
		v_E_inc=vertices_E_inc[evt,tr][incSlice]
		#incEnergies.append(ak.sum(v_E_inc))
		if(len(v_ind_inc)<2):
			continue
		TrNet=net.Network(v_ind_inc,v_x_inc,v_y_inc,v_z_inc,v_E_inc)
		edges_1 = TrNet.edgeBuilderNew(nEdg)
		edges_1 = ak.flatten(edges_1[ak.num(edges_1) > 0].to_list())
		if(optCen=="c_nxkatz"):
			centrality=TrNet.nXCentralityKatz(v_ind_inc,edges_1,isDirected)
		elif(optCen=="c_nxpr"):
			centrality=TrNet.centralityPageRank(v_ind_inc,edges_1,0.85,isDirected)
		elif(optCen=="c_pr"):
			centrality=TrNet.nXCentralityPageRank(v_ind_inc,edges_1,0.85,isDirected)
		elif(optCen=="c_nxeigen"):
			if(len(v_ind_inc)<3):
				continue
			centrality=TrNet.nXCentralityEigen(v_ind_inc,edges_1,isDirected)
			
		adjMatrix=TrNet.adjM(v_ind_inc,edges_1)

		cenProfList=TrNet.centralityProf(adjMatrix,centrality)
		incCenProfArray.append(cenProfList)
		incNVerticesList.append(len(v_ind_inc))


with open(options.output,"ab") as f:
	pickle.dump(incCenProfArray, f)
	pickle.dump(incNVerticesList, f)
	
