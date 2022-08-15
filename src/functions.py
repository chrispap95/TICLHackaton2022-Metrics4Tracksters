import numpy as np
import awkward as ak
import uproot
from pylab import cm
import matplotlib.pyplot as plt
import networkx as nx

def edgeBuilderNew(vertices_indexes, vertices_x, vertices_y, vertices_z, vertices_E, nEdg=1):
    # Remove that exception for the moment. Not sure if needed.
    #if len(vertices_indexes) <= nEdg:
    #    raise ValueError("Number of attempted connections 'nEdg' cannot exceed the size of the graph")
    # Create matrix of indexes
    indexes = np.stack([ak.to_numpy(vertices_indexes)]*len(vertices_indexes),axis=0)
    # Perform energy filtering. Keep only nodes that have greater energy that the specified node
    enMtrx = np.stack([ak.to_numpy(vertices_E)]*len(vertices_E),axis=0)
    energyMask = enMtrx > np.transpose(enMtrx)
    # Calculate euclidean distance between all nodes and apply energy mask
    distMatr = euclideanMatrix(vertices_x, vertices_y, vertices_z)
    # Sort each row and keep the indexes of the sorted arrays
    distSort = np.argsort(distMatr,axis=1)
    # Sort the euclidean distance using the indexes from previous step
    energyMaskSorted = energyMask[np.arange(energyMask.shape[0])[:,None], distSort]
    indexesSorted = indexes[np.arange(energyMask.shape[0])[:,None], distSort]
    # Some awkward magic - converts innermost array length from const to var
    indexesSorted = ak.unflatten(ak.flatten(indexesSorted), ak.num(indexesSorted))
    energyMaskSorted = ak.unflatten(ak.flatten(energyMaskSorted), ak.num(energyMaskSorted))
    # Filter nodes that have lower energy and keep nEdg nearest neighbors
    indexesSorted = indexesSorted[energyMaskSorted]
    indexesSorted = indexesSorted[:,:nEdg]
    return ak.cartesian([vertices_indexes, indexesSorted])

def euclideanMatrix(vertices_x,vertices_y,vertices_z):
    ver_x = ak.to_numpy(vertices_x)
    ver_y = ak.to_numpy(vertices_y)
    ver_z = ak.to_numpy(vertices_z)
    #subtract.outer to compute difference in all combinations
    diff_x = np.subtract.outer(ver_x,ver_x)
    diff_y = np.subtract.outer(ver_y,ver_y)
    diff_z = np.subtract.outer(ver_z,ver_z)
    euclidean_matrix = np.sqrt(diff_x**2+diff_y**2+diff_z**2)
    
    return euclidean_matrix

def edgeBuilderCyclEv(vertices_indexes):
    """
    Builds a cyclical graph for debugging purposes
    input: array of vertices for all events & tracksters
    """
    edges = ak.ArrayBuilder()
    for v_event in vertices_indexes:
        edges.begin_list()
        for v_trackster in v_event:
            edges.begin_list()
            for ind, v_node in enumerate(v_trackster):
                edges.begin_tuple(2)
                edges.index(0).integer(v_node)
                if ind < len(v_trackster)-1:
                    edges.index(1).integer(v_trackster[ind+1])
                else:
                    edges.index(1).integer(v_trackster[0])
                edges.end_tuple()
            edges.end_list()
        edges.end_list()
    return edges

def edgeBuilderCyclTr(vertices_indexes):
    edges = ak.ArrayBuilder()
    for v_trackster in vertices_indexes:
        edges.begin_list()
        for idn, v_node in enumerate(v_trackster):
            edges.begin_tuple(2)
            edges.index(0).integer(v_node)
            if idn < len(v_trackster)-1:
                edges.index(1).integer(v_trackster[idn+1])
            else:
                edges.index(1).integer(v_trackster[0])
            edges.end_tuple()
        edges.end_list()
    return edges

def edgeBuilderNNEv(vertices_indexes, vertices_x, vertices_y, vertices_z, vertices_E):
    edges = ak.ArrayBuilder()
    for ide, v_event in enumerate(vertices_indexes):
        edges.begin_list()
        for idt, v_trackster in enumerate(v_event):
            euMatr = euclideanMatrix(vertices_x[ide,idt],vertices_y[ide,idt],vertices_z[ide,idt])
            edges.begin_list()
            for idn, v_node in enumerate(v_trackster):
                dist_array = euMatr[idn][vertices_E[ide,idt] > vertices_E[ide,idt,idn]]
                if len(dist_array) == 0:
                    continue
                min_val = np.min(dist_array)
                idx = np.where(euMatr[idn] == min_val)[0][0]
                edges.begin_tuple(2)
                edges.index(0).integer(v_node)
                edges.index(1).integer(v_trackster[idx])
                edges.end_tuple()
            edges.end_list()
        edges.end_list()
    return edges

def edgeBuilderNNTr(vertices_indexes, vertices_x, vertices_y, vertices_z, vertices_E):
    edges = ak.ArrayBuilder()
    for idt, v_trackster in enumerate(vertices_indexes):
        euMatr = euclideanMatrix(vertices_x[idt],vertices_y[idt],vertices_z[idt])
        edges.begin_list()
        for idn, v_node in enumerate(v_trackster):
            dist_array = euMatr[idn][vertices_E[idt] > vertices_E[idt,idn]]
            if len(dist_array) == 0:
                continue
            min_val = np.min(dist_array)
            idx = np.where(euMatr[idn] == min_val)[0][0]
            edges.begin_tuple(2)
            edges.index(0).integer(v_node)
            edges.index(1).integer(v_trackster[idx])
            edges.end_tuple()
        edges.end_list()
    return edges

def calcWeight(mode, cluster1, cluster2,n):
    E1 = cluster1.energy()
    E2 = cluster2.energy()
    dx = cluster2.x()-cluster1.x()
    dy = cluster2.y()-cluster1.y()
    dz = cluster2.z()-cluster1.z()
    dist = sqrt(np.power(dx,2)+np.power(dy,2)+np.power(dz,2))
    """
    Default is not weighted
    Method 1: max(E1, E2)
    Method 2: |E1 - E2|
    Method 3: (d)^(-n)
    Method 4: (method 1 or 2)*(method 3)
    """
    weight = 1;
    if(mode==1):
        weight = max(E1, E2)
    elif(mode==2):
        weight = np.abs(E1-E2)
    elif(mode==3):
        weight = np.power(dist,-n)
    elif(mode==4):
        weight = np.abs(E1-E2)*np.power(dist,-n)
        
    return weight

def adjM(nodes,edges,isDirected=False):
    adj = np.zeros((len(nodes),len(nodes)))
    for edge in edges:
        #cluster1=layerCluster(edge[0])
        #cluster2=layerCluster(edge[1])
        #weight=calcWeight(wtMode,cluster1,cluster2,1)
        weight=1
        idx0=np.where(nodes==edge.to_list()[0])
        idx1=np.where(nodes==edge.to_list()[1])
        adj[idx0,idx1] = weight
        if(not isDirected):
            adj[idx1,idx0] = weight
    maxVal=adj.max()
    return adj/maxVal

def centralityEigen(nodes,edges,isDirected=False,printStuff=False):
    adj=adjM(nodes,edges,isDirected)
    rows,columns= adj.shape
    eigvals,vecr= np.linalg.eig(adj)
    i=np.argmax(np.abs(eigvals)) 
    c_eig= vecr[:,i]
    if(c_eig[0]<0):
        c_eig *=-1
    if(printStuff):
        print(eigvals)
        print(vecr)
    c_eig_real=c_eig.real
    norm=np.linalg.norm(c_eig_real)
    return c_eig_real/sum(c_eig_real)

def centralityKatz(nodes,edges,isDirected=False,printStuff=False):
    adj=adjM(nodes,edges,isDirected)
    rows,columns= adj.shape
    Id=np.identity(rows)
    eigvals,vecr= np.linalg.eig(adj)
    i=np.argmax(np.abs(eigvals)) 
    alpha= 0.9/eigvals[i]
    c_katz=(np.linalg.inv(Id-alpha*adj.T)-Id)@np.ones((rows)).T
    if(printStuff):
        print(eigvals)
        print(vecr)
    c_katz_real=c_katz.real
    norm=np.linalg.norm(c_katz_real)
    return c_katz_real/sum(c_katz_real)
    
def centralityPageRank(nodes,edges,df,isDirected=False,printStuff=False):
    adj=adjM(nodes,edges,isDirected)
    rows,columns= adj.shape
    m_ones=np.ones((rows,columns))
    m_pr=df*adj+(1-df)*m_ones/rows
    eigvals,vecr= np.linalg.eig(m_pr)
    i=np.argmax(np.abs(eigvals)) 
    c_pr= vecr[:,i]
    if(c_pr[0]<0):
        c_pr *=-1
    if(printStuff):
        print(eigvals)
        print(vecr)
    c_pr_real=c_pr.real
    norm=np.linalg.norm(c_pr_real)
    return c_pr_real/sum(c_pr_real)

def nXCentralityEigen(nodes,edges,isDirected=False):
    if(isDirected):
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_edges_from(ak.to_numpy(edges))
    G.add_nodes_from(ak.to_numpy(nodes))
    centr_d = nx.eigenvector_centrality_numpy(G)
    centr_np = np.array(list(centr_d.items()))
    return centr_np[centr_np[:, 0].argsort()][:,1]

def nXCentralityKatz(nodes,edges,isDirected=False):
    if(isDirected):
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_edges_from(ak.to_numpy(edges))
    G.add_nodes_from(ak.to_numpy(nodes))
    centr_d = nx.katz_centrality_numpy(G)
    centr_np = np.array(list(centr_d.items()))
    centr_f= centr_np[centr_np[:, 0].argsort()][:,1]
    return centr_f/sum(centr_f)

def nXCentralityPageRank(nodes,edges,isDirected=False):
    if(isDirected):
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_edges_from(ak.to_numpy(edges))
    G.add_nodes_from(ak.to_numpy(nodes))
    centr_d = nx.pagerank(G,0.85)
    centr_np = np.array(list(centr_d.items()))
    centr_f= centr_np[centr_np[:, 0].argsort()][:,1]
    return centr_f/sum(centr_f)

def longestPathSource(nodes,edges,centralities,isDirected=False):
    """
    Finds the longest path in the network from the max
    of the stortest path algorithm.
    """
    if(isDirected):
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_edges_from(ak.to_numpy(edges))
    G.add_nodes_from(ak.to_numpy(nodes))
    
    #Highest centrality node
    i_centralityMax=np.argmax(centralities)
    source=nodes[i_centralityMax]
    #Finds the shortest path from the highest centrality to all other nodes
    pathList=nx.shortest_path_length(G,source=source)
    #Takes the max of all paths to find the longest path
    longestShortestPath=max(pathList.values())
    return longestShortestPath

def longestPathInitialNode(nodes,edges,isDirected=False):
    """
    Finds the longest path in the network from the max
    of the stortest path algorithm.
    """
    if(isDirected):
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_edges_from(ak.to_numpy(edges))
    G.add_nodes_from(ak.to_numpy(nodes))
    
    source=nodes[0]
    #Finds the shortest path from the first node to all other nodes
    pathList=nx.shortest_path_length(G,source=source)
    #Takes the max of all paths to find the longest path
    longestShortestPath=max(pathList.values())
    return longestShortestPath

def plotTrackster(fig, ax, x, y, z, heatmap=None, indexes=None, edges=None, label='Vertex Energy (GeV)'):
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    colmap = cm.ScalarMappable(cmap=cm.viridis)
    if len(heatmap) > 0 :
        colmap.set_array(heatmap)
        yg = ax.scatter(x, y, z, c=cm.viridis(heatmap/max(heatmap)), marker='o', linewidth=2)
        cb = fig.colorbar(colmap,label=label)
    else:
        yg =ax.scatter(x, y, z, marker='o') 
    edges=ak.to_numpy(edges)
    if len(heatmap) > 0:
        for ind in edges:
            if len(ind) == 0:
                continue
            for ied in ind:
                idx0 = ak.where(indexes == ied.to_list()[0])[0][0]
                idx1 = ak.where(indexes == ied.to_list()[1])[0][0]
                ax.plot(
                    [x[idx0] ,x[idx1]],
                    [y[idx0] ,y[idx1]],
                    [z[idx0] ,z[idx1]],
                    'black'
                )
    plt.show()


def incompleteTracksters(vertices_layer,mean,std,seed1=None,seed2=None):
    v_layer=ak.to_numpy(vertices_layer)
    n=len(vertices_layer)
    #print(n)
    if(seed1!=None):
        np.random.seed(seed1)
    q1=np.random.normal(mean,std)
    if(seed2!=None):
        np.random.seed(seed2)
    q2=np.random.normal(mean,std)
    return slice(int(n*q1),int(n*(1-q2)))

def ld(vertices_z,vertices_E):
    if(vertices_z[0]<0):
        offset=320
    else:
        offset=-320
    
    ldVal=sum((vertices_z+offset)*vertices_E)
    return ldVal

def delta_Eta(vertices_y,vertices_z,barycenter_eta):
    theta=np.arctan(vertices_z/vertices_y)
    eta=-np.log(np.tan(theta/2))
    
    return eta-barycenter_eta

def delta_phi(vertices_x,vertices_y,barycenter_phi):
    phi=np.tan(vertices_y/vertices_x)
    
    return phi-barycenter_phi

def delta_eta_phi(vertices_x,vertices_y,vertices_z,barycenter_eta,barycenter_phi):
    deltaPhi=delta_phi(vertices_x,vertices_y,barycenter_phi)
    deltaEta=delta_Eta(vertices_y,vertices_z,barycenter_eta)
    
    return np.mean(np.sqrt(deltaPhi**2+deltaEta**2))

def delta_RT(vertices_x,vertices_y,vertices_E,Eweighted=False):
    argmax_E=ak.argmax(vertices_E)
                       
    vEmax_x=vertices_x[argmax_E]
    vEmax_y=vertices_y[argmax_E]
    RvEmax=np.sqrt(vEmax_x**2+vEmax_y**2)
    
    R=np.sqrt(vertices_x**2+vertices_y**2)
    
    if(Eweighted):
        delta_R=sum((R-RvEmax)**2*vertices_E/vertices_E[argmax_E])
    else:
        delta_R=sum((R-RvEmax)**2)
        
    return delta_R

def delta_R(vertices_x,vertices_y,vertices_z,vertices_E,Eweighted=False):
    argmax_E=ak.argmax(vertices_E)
                       
    vEmax_x=vertices_x[argmax_E]
    vEmax_y=vertices_y[argmax_E]
    vEmax_z=vertices_z[argmax_E]
    RvEmax=np.sqrt(vEmax_x**2+vEmax_y**2+vEmax_z**2)
    
    R=np.sqrt(vertices_x**2+vertices_y**2+vertices_z**2)
    
    delta_R=sum((R-RvEmax)**2)
        
    return delta_R

def delta_RT_std(vertices_x,vertices_y,vertices_E,Eweighted=False):
    argmax_E=ak.argmax(vertices_E)
                       
    vEmax_x=vertices_x[argmax_E]
    vEmax_y=vertices_y[argmax_E]
    RvEmax=np.sqrt(vEmax_x**2+vEmax_y**2)
    
    R=np.sqrt((vertices_x-vEmax_x)**2+(vertices_y-vEmax_y)**2)

    delta_R_std=np.sqrt(np.abs(sum(R**2)-sum(R)**2))
        
    return delta_R_std
                       
def maxE_z(vertices_z,vertices_E):
    argmax_E=ak.argmax(vertices_E)
    if(vertices_z[0]<0):
        maxE_z=-1*(vertices_z[argmax_E]+320)
    else:
        maxE_z=(vertices_z[argmax_E]-320)
    return maxE_z
    
def sd(vertices_z,vertices_E):
    ldVal=ld(vertices_z,vertices_E)
    E_tot=ak.sum(vertices_E)
    return ldVal/E_tot

def maxAbsZ(vertices_z):
    if(vertices_z[0]<0):
        maxZ=ak.min(vertices_z)
    else:
        maxZ=ak.max(vertices_z)
    return abs(maxZ)


