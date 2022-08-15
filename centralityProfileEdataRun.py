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

print(type(options.Ntr))
N=int(options.Ntr)
nEdg=int(options.nEdg)
optCen=options.centrality
isDirected=options.isDirected

filename = "root://eosuser.cern.ch//eos/user/d/dmagdali/tracksters_ds_100e.root"
file_100e = uproot.open(filename)
tracksters = file_100e['tracksters']

vertices_indexes = tracksters['vertices_indexes'].array()
vertices_x = tracksters['vertices_x'].array()
vertices_y = tracksters['vertices_y'].array()
vertices_z = tracksters['vertices_z'].array()
vertices_E = tracksters['vertices_energy'].array()
trackster_label = tracksters['trackster_label'].array()

complete = trackster_label == 1
incomplete = trackster_label == 0

vertices_indexes_inc = vertices_indexes[incomplete]
vertices_x_inc = vertices_x[incomplete]
vertices_y_inc = vertices_y[incomplete]
vertices_z_inc = vertices_z[incomplete]
vertices_E_inc = vertices_E[incomplete]

vertices_indexes_com = vertices_indexes[complete]
vertices_x_com = vertices_x[complete]
vertices_y_com = vertices_y[complete]
vertices_z_com = vertices_z[complete]
vertices_E_com = vertices_E[complete]

tooSmall_inc = ak.num(vertices_indexes_inc,axis=-1) > 2
tooSmall_com = ak.num(vertices_indexes_com,axis=-1) > 2

vertices_indexes_inc_g = vertices_indexes_inc[tooSmall_inc]
vertices_x_inc_g = vertices_x_inc[tooSmall_inc]
vertices_y_inc_g = vertices_y_inc[tooSmall_inc]
vertices_z_inc_g = vertices_z_inc[tooSmall_inc]
vertices_E_inc_g = vertices_E_inc[tooSmall_inc]

vertices_indexes_com_g = vertices_indexes_com[tooSmall_com]
vertices_x_com_g = vertices_x_com[tooSmall_com]
vertices_y_com_g = vertices_y_com[tooSmall_com]
vertices_z_com_g = vertices_z_com[tooSmall_com]
vertices_E_com_g = vertices_E_com[tooSmall_com]

comCenProfArray=[]
comNVerticesList=[]

for tr in range(N):
	print(tr)
	#v_layers=vertices_layers[evt][tr]
	v_ind=vertices_indexes_com_g[tr]
	#v_x=vertices_x[evt][tr]
	#v_y=vertices_y[evt][tr]
	#v_z=vertices_z[evt][tr]
	#v_E=vertices_E[evt][tr]
	TrNet=net.Network(v_ind,vertices_x_com_g[tr],vertices_y_com_g[tr],
							 vertices_z_com_g[tr],vertices_E_com_g[tr])
	edges_1 = TrNet.edgeBuilderNew(nEdg)
	edges_1 = ak.flatten(edges_1[ak.num(edges_1) > 0].to_list())
	if(optCen=="c_nxkatz"):
		centrality=TrNet.nXCentralityKatz(v_ind,edges_1,isDirected)
	elif(optCen=="c_nxpr"):
		centrality=TrNet.centralityPageRank(v_ind,edges_1,0.85,isDirected)
	elif(optCen=="c_pr"):
		centrality=TrNet.nXCentralityPageRank(v_ind,edges_1,0.85,isDirected)
	elif(optCen=="c_nxeigen"):
		centrality=TrNet.nXCentralityEigen(v_ind,edges_1,isDirected)
		
	adjMatrix=TrNet.adjM(v_ind,edges_1)

	cenProfList=TrNet.centralityProf(adjMatrix,centrality)
	comCenProfArray.append(cenProfList)
	comNVerticesList.append(len(v_ind))

with open(options.output,"wb") as f:
	pickle.dump(comCenProfArray, f)
	pickle.dump(comNVerticesList, f)

incCenProfArray=[]
incNVerticesList=[]
#incEnergies=[]
for tr in range(N):
	v_ind_inc=vertices_indexes_inc_g[tr]
	v_x_inc=vertices_x_inc_g[tr]
	v_y_inc=vertices_y_inc_g[tr]
	v_z_inc=vertices_z_inc_g[tr]
	v_E_inc=vertices_E_inc_g[tr]
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
		centrality=TrNet.nXCentralityEigen(v_ind_inc,edges_1,isDirected)
		
	adjMatrix=TrNet.adjM(v_ind_inc,edges_1)

	cenProfList=TrNet.centralityProf(adjMatrix,centrality)
	incCenProfArray.append(cenProfList)
	incNVerticesList.append(len(v_ind_inc))


with open(options.output,"ab") as f:
	
	pickle.dump(incCenProfArray, f)
	pickle.dump(incNVerticesList, f)
	
