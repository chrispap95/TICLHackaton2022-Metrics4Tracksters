import numpy as np
import awkward as ak
import uproot
from pylab import cm
import matplotlib.pyplot as plt
import networkx as nx

class Network:
    def __init__(self, vertices_indexes, vertices_x,vertices_y,vertices_z,vertices_E):
        self.ind = vertices_indexes
        self.x = vertices_x
        self.y = vertices_y
        self.z = vertices_z
        self.E = vertices_E
        self.wtMode=0
        self.weights=np.empty(len(vertices_indexes))
        
    def setWtMode(self,wtModeVal):
        self.wtMode=wtModeVal
        
    def edgeBuilderNew(self,nEdg=1):
        # Remove that exception for the moment. Not sure if needed.
        #if len(vertices_indexes) <= nEdg:
        #    raise ValueError("Number of attempted connections 'nEdg' cannot exceed the size of the graph")
        # Create matrix of indexes
        indexes = np.stack([ak.to_numpy(self.ind)]*len(self.ind),axis=0)
        # Perform energy filtering. Keep only nodes that have greater energy that the specified node
        enMtrx = np.stack([ak.to_numpy(self.E)]*len(self.E),axis=0)
        energyMask = enMtrx > np.transpose(enMtrx)
        # Calculate euclidean distance between all nodes and apply energy mask
        distMatr = self.euclideanMatrix()
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
        return ak.cartesian([self.ind, indexesSorted])

    def euclideanMatrix(self):
        ver_x = ak.to_numpy(self.x)
        ver_y = ak.to_numpy(self.y)
        ver_z = ak.to_numpy(self.z)
        #subtract.outer to compute difference in all combinations
        diff_x = np.subtract.outer(ver_x,ver_x)
        diff_y = np.subtract.outer(ver_y,ver_y)
        diff_z = np.subtract.outer(ver_z,ver_z)
        euclidean_matrix = np.sqrt(diff_x**2+diff_y**2+diff_z**2)

        return euclidean_matrix

    def calcWeight(self, id1, id2,n):
        E1 = self.E[id1]
        E2 = self.E[id2]
        dx = self.x[id2]-self.x[id1]
        dy = self.y[id2]-self.y[id1]
        dz = self.z[id2]-self.z[id1]
        dist = np.sqrt(np.power(dx,2)+np.power(dy,2)+np.power(dz,2))
        """
        Default is not weighted
        Method 1: max(E1, E2)
        Method 2: |E1 - E2|
        Method 3: (d)^(-n)
        Method 4: (method 1 or 2)*(method 3)
        """
        weight = 1;
        if(self.wtMode==1):
            weight = max(E1, E2)
        elif(self.wtMode==2):
            weight = np.abs(E1-E2)
        elif(self.wtMode==3):
            weight = np.power(dist,-n)
        elif(self.wtMode==4):
            weight = np.abs(E1-E2)*np.power(dist,-n)
        elif(self.wtMode==5):
            weight = E2
        return weight
    
    def calcWeights(self,nodes,edges,n):
        for edge in edges:
            #cluster1=layerCluster(edge[0])
            #cluster2=layerCluster(edge[1])
            
            #weight=1
            id1=np.where(nodes==edge.to_list()[0])
            id2=np.where(nodes==edge.to_list()[1])
            E1 = self.E[id1]
            E2 = self.E[id2]
            dx = self.x[id2]-self.x[id1]
            dy = self.y[id2]-self.y[id1]
            dz = self.z[id2]-self.z[id1]
            dist = np.sqrt(np.power(dx,2)+np.power(dy,2)+np.power(dz,2))
            """
            Default is not weighted
            Method 1: max(E1, E2)
            Method 2: |E1 - E2|
            Method 3: (d)^(-n)
            Method 4: (method 1 or 2)*(method 3)
            """
            weight = 1;
            if(self.wtMode==1):
                weight = max(E1, E2)
            elif(self.wtMode==2):
                weight = np.abs(E1-E2)
            elif(self.wtMode==3):
                weight = np.power(dist,-n)
            elif(self.wtMode==4):
                weight = np.abs(E1-E2)*np.power(dist,-n)
            elif(self.wtMode==5):
                weight = E2
            
            self.weights[id1]=weight

    def adjM(self,nodes,edges,isDirected=False):
        adj = np.zeros((len(nodes),len(nodes)))
        for edge in edges:
            #cluster1=layerCluster(edge[0])
            #cluster2=layerCluster(edge[1])
            
            #weight=1
            idx0=np.where(nodes==edge.to_list()[0])
            idx1=np.where(nodes==edge.to_list()[1])
            weight=self.calcWeight(idx0,idx1,1)
            adj[idx0,idx1] = weight
            self.weights[idx0]=weight
            if(not isDirected):
                adj[idx1,idx0] = weight
        maxVal=adj.max()
        return adj/maxVal

    def centralityEigen(self,nodes,edges,isDirected=False,printStuff=False):
        adj=self.adjM(nodes,edges,isDirected)
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

    def centralityKatz(self,nodes,edges,isDirected=False,printStuff=False):
        adj=self.adjM(nodes,edges,isDirected)
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

    def centralityPageRank(self,nodes,edges,df,isDirected=False,printStuff=False):
        adj=self.adjM(nodes,edges,isDirected)
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

    def nXCentralityEigen(self,nodes,edges,isDirected=False):
        if(isDirected):
            G=nx.DiGraph()
        else:
            G=nx.Graph()
        G.add_edges_from(ak.to_numpy(edges),weights=self.weights)
        G.add_nodes_from(ak.to_numpy(nodes),weights=self.weights)
        centr_d = nx.eigenvector_centrality(G)
        centr_np = np.array(list(centr_d.items()))
        return centr_np[centr_np[:, 0].argsort()][:,1]

    def nXCentralityKatz(self,nodes,edges,isDirected=False):
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

    def nXCentralityPageRank(self,nodes,edges,df,isDirected=False):
        if(isDirected):
            G=nx.DiGraph()
        else:
            G=nx.Graph()
        G.add_edges_from(ak.to_numpy(edges))
        G.add_nodes_from(ak.to_numpy(nodes))
        centr_d = nx.pagerank(G,df)
        centr_np = np.array(list(centr_d.items()))
        centr_f= centr_np[centr_np[:, 0].argsort()][:,1]
        return centr_f/sum(centr_f)

    def longestPathSource(self,nodes,edges,centralities,isDirected=False):
        """
        Finds the longest path in the network from the max
        of the stortest path algorithm.
        """
        if(isDirected):
            G=nx.DiGraph()
        else:
            G=nx.Graph()
        G.add_edges_from(ak.to_numpy(edges),weights=self.weights)
        G.add_nodes_from(ak.to_numpy(nodes),weights=self.weights)

        #Highest centrality node
        i_centralityMax=np.argmax(centralities)
        source=nodes[i_centralityMax]
        #Finds the shortest path from the highest centrality to all other nodes
        pathList=nx.shortest_path_length(G,source=source)
        #Takes the max of all paths to find the longest path
        longestShortestPath=max(pathList.values())
        return longestShortestPath

    def longestPathInitialNode(self,nodes,edges,isDirected=False):
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

    def plotTrackster(self,fig, ax, x, y, z, heatmap=None, indexes=None, edges=None, label='Vertex Energy (GeV)'):
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


    def incompleteTracksters(self,vertices_layer,seed1=None,seed2=None):
        v_layer=ak.to_numpy(vertices_layer)
        n=len(vertices_layer)
        #print(n)
        if(seed1!=None):
            np.random.seed(seed1)
        q1=np.random.normal(0.18,0.02)
        if(seed2!=None):
            np.random.seed(seed2)
        q2=np.random.normal(0.18,0.02)
        return slice(int(n*q1),int(n*(1-q2)))

    def ld(self):
        if(self.z[0]<0):
            offset=320
        else:
            offset=-320

        ldVal=sum((self.z+offset)*self.E)
        return ldVal

    def delta_RT(self,Eweighted=False):
        argmax_E=ak.argmax(self.E)

        vEmax_x=self.x[argmax_E]
        vEmax_y=self.y[argmax_E]
        RvEmax=np.sqrt(vEmax_x**2+vEmax_y**2)

        R=np.sqrt(self.x**2+self.y**2)

        if(Eweighted):
            delta_R=sum((R-RvEmax)**2*self.E/self.E[argmax_E])
        else:
            delta_R=sum((R-RvEmax)**2)

        return delta_R

    def delta_R(self,Eweighted=False):
        argmax_E=ak.argmax(self.E)

        vEmax_x=self.x[argmax_E]
        vEmax_y=self.y[argmax_E]
        vEmax_z=self.z[argmax_E]
        RvEmax=np.sqrt(vEmax_x**2+vEmax_y**2+vEmax_z**2)

        R=np.sqrt(self.x**2+self.y**2+self.z**2)

        delta_R=sum((R-RvEmax)**2)

        return delta_R

    def delta_RT_std(self,Eweighted=False):
        argmax_E=ak.argmax(self.E)

        vEmax_x=self.x[argmax_E]
        vEmax_y=self.y[argmax_E]
        RvEmax=np.sqrt(vEmax_x**2+vEmax_y**2)

        R=np.sqrt((self.x-vEmax_x)**2+(selfy-vEmax_y)**2)

        delta_R_std=np.sqrt(np.abs(sum(R**2)-sum(R)**2))

        return delta_R_std

    def maxE_z(self):
        argmax_E=ak.argmax(self.E)
        if(self.z[0]<0):
            maxE_z=-1*(self.z[argmax_E]+320)
        else:
            maxE_z=(self.z[argmax_E]-320)
        return maxE_z

    def sd(self):
        ldVal=self.ld()
        E_tot=ak.sum(self.E)
        return ldVal/E_tot

    def maxAbsZ(self):
        if(self.z[0]<0):
            maxZ=ak.min(self.z)
        else:
            maxZ=ak.max(self.z)
        return abs(maxZ)
    
    
    def centralityProfIter(self,neighborsList,centrality,adjMatrix):
        for i in range(len(neighborsList)):
            if(neighborsList[i]==0):
                continue
            if(i in self.doneList):
                continue
            else:
                #print(centrality)
                self.sumCen+=centrality[i]
                self.n+=1
                self.doneList.append(i)
                self.nextList.append(adjMatrix[i,:])
                #print(nextList)

    def centralityProf(self,adjMatrix,centrality):
        self.doneList=[]
        self.nextList=[]
        cenProfList=[max(centrality)]
        i_cen_max=np.argmax(centrality)
        firstList=adjMatrix[i_cen_max]
        self.sumCen=0
        self.n=0
        self.centralityProfIter(firstList,centrality,adjMatrix)
        cenProfList.append(self.sumCen/self.n)
        #print(firstList)
        #nextList=adjMatrix[firstList==1.]
        #print(nextList)
        while(len(self.doneList)<adjMatrix.shape[0]):
            loopList=self.nextList
            self.nextList=[]
            #print(j)
            self.sumCen=0
            self.n=0
            for i in loopList:
                #print("loopList= {}".format(loopList))
                #print("i= {}".format(i))
                self.centralityProfIter(i,centrality,adjMatrix)
                #print("doneList= {}".format(self.doneList))
                #print(i==1.)

            if self.n!=0:
                cenProfList.append(self.sumCen/self.n)
            #print(loopList==1.)
            #nextList=adjMatrix[loopList==1.]
            
        return cenProfList

    def energyProfIter(self,neighborsList,adjMatrix):
        for i in range(len(neighborsList)):
            if(neighborsList[i]==0):
                continue
            if(i in self.doneList):
                continue
            else:
                self.sumE+=self.E[i]/sum(self.E)
                self.n+=1
                self.doneList.append(i)
                #print(adjMatrix[i,:])
                self.nextList.append(adjMatrix[i,:])

    
    def energyProf(self,adjMatrix):
        self.nextList=[]
        energyProfList=[max(self.E)/sum(self.E)]
        i_E_max=np.argmax(self.E)
        self.doneList=[i_E_max]
        firstList=adjMatrix[i_E_max]
        self.sumE=0
        self.n=0
        self.energyProfIter(firstList,adjMatrix)
        energyProfList.append(self.sumE/self.n)
        while(len(self.doneList)<adjMatrix.shape[0]):
            loopList=self.nextList
            self.nextList=[]
            #print(j)
            self.sumE=0
            self.n=0
            for i in loopList:
                self.energyProfIter(i,adjMatrix)

            if self.n!=0:
                energyProfList.append(self.sumE/self.n)
            #print(loopList==1.)
            #nextList=adjMatrix[loopList==1.]
            
        return energyProfList
