import numpy as np
import awkward as ak
import uproot

def euclideanMatrix(vertices_x,vertices_y,vertices_z):
    ver_x = ak.to_numpy(vertices_x[0,0])
    ver_y = ak.to_numpy(vertices_y[0,0])
    ver_z = ak.to_numpy(vertices_z[0,0])
    #subtract.outer to compute difference in all combinations
    diff_x = np.subtract.outer(ver_x,ver_x)
    diff_y = np.subtract.outer(ver_y,ver_y)
    diff_z = np.subtract.outer(ver_z,ver_z)
    euclidean_matrix = np.sqrt(diff_x**2+diff_y**2+diff_z**2)
    
    return euclidean_matrix
    

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
        idx0=np.where(nodes==edge[0])
        idx1=np.where(nodes==edge[1])
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
    norm=np.linalg.norm(c_eig)
    return c_eig/norm

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
    norm=np.linalg.norm(c_katz)
    return c_katz/norm
    
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
    norm=np.linalg.norm(c_pr)
    return c_pr/norm
    
