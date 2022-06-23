import numpy as np
import awkward as ak
import uproot
from pylab import cm
import matplotlib.pyplot as plt
import networkx as nx

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

def centralityEigen(adj,printStuff=False):
    rows,columns= adj.shape
    eigvals,vecl= np.linalg.eig(adj)
    i=np.argmax(np.abs(eigvals)) 
    c_eig= vecl[:,i]
    if(c_eig[0]<0):
        c_eig *=-1
    if(printStuff):
        print(eigvals)
        print(vecl)
    c_eig_real=c_eig.real
    norm=np.linalg.norm(c_eig_real)
    return c_eig_real/norm

def centralityKatz(adj,printStuff=False):
    rows,columns= adj.shape
    Id=np.identity(rows)
    eigvals,vecl= np.linalg.eig(adj)
    i=np.argmax(np.abs(eigvals)) 
    alpha= 0.9/eigvals[i]
    c_katz=(np.linalg.inv(Id-alpha@adj.T)-Id)@np.ones((rows)).T
    if(printStuff):
        print(eigvals)
        print(vecl)
    c_katz_real=c_katz.real
    norm=np.linalg.norm(c_katz_real)
    return c_katz_real/norm
    
def centralityPageRank(adj,df,printStuff=False):
    rows,columns= adj.shape
    m_ones=np.ones((rows,columns))
    m_pr=df*adj+(1-df)*m_ones/rows
    eigvals,vecl= np.linalg.eig(m_pr)
    i=np.argmax(np.abs(eigvals)) 
    c_pr= vecl[:,i]
    if(c_pr[0]<0):
        c_pr *=-1
    if(printStuff):
        print(eigvals)
        print(vecl)
    c_pr_real=c_pr.real
    norm=np.linalg.norm(c_pr_real)
    return c_pr_real/norm

def nXCentralityEigen(nodes,edges):
    G=nx.Graph()
    G.add_edges_from(ak.to_numpy(edges))
    G.add_nodes_from(ak.to_numpy(nodes))
    return nx.eigenvector_centrality_numpy(G)

def nXCentralityKatz(nodes,edges):
    G=nx.Graph()
    G.add_edges_from(ak.to_numpy(edges))
    G.add_nodes_from(ak.to_numpy(nodes))
    return nx.katz_centrality_numpy(G)
    
def plotTrackster(fig,ax,vertices_x,vertices_y,vertices_z,heatmapVals,plotOption='heatmap',heatmapLabel='None'):
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if(plotOption=='heatmap'):
        colmap = cm.ScalarMappable(cmap=cm.viridis)
        colmap.set_array(heatmapVals)
        yg = ax.scatter(vertices_x, vertices_y, vertices_z, c=cm.viridis(heatmapVals/max(heatmapVals)), marker='o')
        if(heatmapLabel=='None'):
            cb = fig.colorbar(colmap)
        cb = fig.colorbar(colmap,label=heatmapLabel)

    if(plotOption=='position'):
        yg =ax.scatter(vertices_x, vertices_y, vertices_z, marker='o')
    
    plt.show()
