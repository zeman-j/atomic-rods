# -*- coding: utf-8 -*-
"""
Numerical approximation of the relaxed quadratic form in the model for ultrathin 
elastic rods from [1] (see equation (3.3)), which is inspired by nanowires.
The implementation in its current version is only illustrative and not optimized 
for best performance on medium or large cross sections of the rod.

The cell energy function originates from a mass-spring model with 
nearest-neighbour and next-to-nearest-neighbour interactions on a cubic lattice 
(see [1, Example 2.1]).

To obtain the value of the relaxed quadratic form at a given 
skew-symmetric matrix A, minimization over variables called alpha and g is performed. 
Alpha is related to warping of the rod's cross section in torsion and lateral 
contraction due to bending, whereas g is a stretch vector, uniform
in the cross-sectional plane.

The minimizer is visualized using 3D plotting in Matplotlib.

For higher physical relevance, surface temrs would also have to be added 
to the energy functional J, depending on the chosen cross section shape.

----------
References

[1] B. Schmidt and J. Zeman. A bending-torsion theory for thin
and ultrathin rods as a Γ-limit of atomistic models, 2022.
arXiv:2208.04199.

@author: Jiri Zeman
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#positive stiffnesses of springs connecting atoms, feel free to change these values
K14 = 1.0 #$K_1/4$ in [1, Example 2.1]
K24 = 1.0 #$K_2/4$

#tunable (real-valued) parameters that influence the result, feel free to change their values
#the actual input of this program
curv2=0.5 #$\kappa_2$ in [1, Proposition 3.2]
curv3=0.0 #$\kappa_3$
torsion=0.0 #$\tau$

#lattice of cross-sectional midpoints, feel free to change (add or remove points)
#$\mathcal{L}'$ in [1]
Lprime =np.array([[0.5,0.5],[0.5,1.5],[1.5,0.5],[1.5,1.5]])

#another tested choice, although a bit more computationally expensive
#Lprime =np.array([[0.5,0.5],[0.5,1.5],[1.5,0.5],[1.5,1.5],[2.5,0.5],[2.5,1.5],[-0.5,0.5],[-0.5,1.5],[0.5,2.5],[1.5,2.5],[0.5,-0.5],[1.5,-0.5]])

#2D direction vectors
z = 0.5*np.array([[-1.0,-1.0],
              [-1.0,1.0],
              [1.0,1.0],
              [1.0,-1.0]])#M could be generated from z: the left submatrix is -2*np.transpose(z)
#the i-th vector is z[i]
M = [[1.0,1.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0],
     [1.0,-1.0,-1.0,1.0,-1.0,1.0,1.0,-1.0]]

#field of neighbours (corners of lattice squares with midpoints in Lprime)
neighbours = np.empty((Lprime.shape[0],4),dtype=np.int32)

#cross-sectional atomic lattice
L=np.empty((0,2))#initialize as an empty list of 2D points
count = 0
#find minima and maxima in x to set bounds for the plot later
minY, maxY = 0.0, 1.0
minZ, maxZ = 0.0, 1.0
for i in range(Lprime.shape[0]):
    x = Lprime[i]
    minY = min(minY,x[0]-0.5)
    minZ = min(minY,x[1]-0.5)
    maxY = max(maxY,x[0]+0.5)
    maxZ = max(maxZ,x[1]+0.5)
    for j in range(4):
        foundInL = (L==(x+z[j])).all(1)
        if foundInL.any():
            neighbours[i,j] = foundInL.nonzero()[0][0]
        else: #if x+zj is not yet in in L, add it
            L = np.append(L,np.expand_dims(x+z[j],axis=0),axis=0)
            neighbours[i,j] = count
            count = count+1

def AfromVector(curv2,curv3,torsion):
    """
    Creates a skew-symmetric matrix from 3 scalar parameters.

    Parameters
    ----------
    curv2 : $\kappa_2$
    curv3 : $\kappa_3$
    torsion : $\tau$

    Returns
    -------
    A : the resulting skew-symmetric 3 by 3 matrix

    """
    A = np.array([[0.0,-curv2,-curv3],
                 [curv2,0.0,-torsion],
                 [curv3,torsion,0.0]])
    return A

def Qcell(H):
    """
    The quadratic form associated with the 2nd derivative of $W_{cell}$ (at the discrete identity)
    given by a mass-spring model with NN and NNN interactions.

    Parameters
    ----------
    H : a 3 by 8 matrix (numpy.ndarray)

    Returns
    -------
    scalar value of the quadratic form.

    """
    return K14*((H[2,1]-H[2,0])**2+(H[1,2]-H[1,1])**2+(H[2,3]-H[2,2])**2+(H[1,0]-H[1,3])**2\
                +(H[0,4]-H[0,0])**2+(H[0,5]-H[0,1])**2+(H[0,6]-H[0,2])**2+(H[0,7]-H[0,3])**2\
                    +(H[2,5]-H[2,4])**2+(H[1,6]-H[1,5])**2+(H[2,7]-H[2,6])**2+(H[1,4]-H[1,7])**2)\
        +K24*((H[1,2]-H[1,0]+H[2,2]-H[2,0])**2+(H[1,1]-H[1,3]+H[2,3]-H[2,1])**2\
              +(H[0,5]-H[0,0]+H[2,5]-H[2,0])**2+(H[0,4]-H[0,1]+H[2,1]-H[2,4])**2\
                  +(H[0,7]-H[0,0]+H[1,7]-H[1,0])**2+(H[0,3]-H[0,4]+H[1,4]-H[1,3])**2\
                      +(H[0,6]-H[0,1]+H[1,6]-H[1,1])**2+(H[0,2]-H[0,5]+H[1,5]-H[1,2])**2\
                          +(H[0,6]-H[0,3]+H[2,6]-H[2,3])**2+(H[0,2]-H[0,7]+H[2,7]-H[2,2])**2\
                              +(H[1,6]-H[1,4]+H[2,6]-H[2,4])**2+(H[1,5]-H[1,7]+H[2,7]-H[2,5])**2)
def J(X,A):
    """
    The cross-section interaction energy functional to be minimized.

    Parameters
    ----------
    X : the values of $\alpha$ at atomic points and $g$, all concatenated into a 15D vector.
    A : the skew-symmetric 3 by 3 strain matrix

    Returns
    -------
    scalar value of the interaction energy

    """
    g = X[-3:]#g is saved in the last three values of X
    alpha = np.empty((3,L.shape[0]))
    for i in range(L.shape[0]):
        alpha[:,i]=X[3*i:3*i+3]
    #now alpha is a very wide matrix, whose columns correspond to values of $\alpha$ at atomic points
    value = 0.0
    thinCorr =0.25*np.dot(A[:,1:],M)
    g38 = np.outer(g,[-0.5,-0.5,-0.5,-0.5,0.5,0.5,0.5,0.5]) 
    for i in range(Lprime.shape[0]):
        alpha38 = np.empty((3,8))
        x = Lprime[i]
        for j in range(4):
            alpha38[:,j] = alpha[:,neighbours[i,j]]
            alpha38[:,j+4] = alpha38[:,j]
        value = value+Qcell(np.outer(np.dot(A[:,1:],x),[-0.5,-0.5,-0.5,-0.5,0.5,0.5,0.5,0.5])
                            +g38+thinCorr+alpha38)
    return value

#initial guess for minimization
alpha0 = np.zeros((3,L.shape[0]))
g0=np.zeros(3)

X0 = np.empty((3*L.shape[0]+3))
X0[-3:] = g0
for i in range(L.shape[0]):
    X0[3*i:3*i+3]=alpha0[:,i]

A=AfromVector(curv2,curv3,torsion)
#J(X0,A)
#including adaptivity seems to pay off in this higher-dimensional problem
#the maximum number of iterations had to be increased
result = minimize(J, X0, method='nelder-mead',args=(A,),options={'xatol': 1e-8, 'disp': True, 'return_all': True,
                                                                 'maxiter':500000,'maxfev':500000,'adaptive':True})
# f = open("log.txt",'w')
# f.write(str(result))
# f.close()
print("Success?", result.success)
Js = [J(X,A) for X in result.allvecs]
N = np.linspace(1,len(result.allvecs),len(result.allvecs))

#plot a convergence curve
plt.plot(N,Js,'-')
plt.xlabel("iteration number")
plt.ylabel("J")
plt.show()

X = result.x
g = X[-3:].reshape(3,1)
alpha = np.empty((3,L.shape[0]))
for i in range(L.shape[0]):
    alpha[:,i]=X[3*i:3*i+3]
print("alpha =",alpha)
print("g =",g)

def plotResult(quantity="alpha"):
    """
    Parameters
    ----------
    quantity : the quantity to plot ("alpha", "g", "A" or "thinCorr")
        The default is "alpha".

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    #plot the reference atomic cubes
    for x in L:
        ax.scatter(x[0],x[1],0.0,marker='o',c='red')
        refPts = ax.scatter(x[0],x[1],1.0,marker='o',c='red')   
        
    if quantity == "g":
        for i in range(L.shape[0]):
            x = L[i]
            defPts = ax.scatter(x[0]-0.5*g[1],x[1]-0.5*g[2],0.0-0.5*g[0],marker='o',c='brown')
            ax.scatter(x[0]+0.5*g[1],x[1]+0.5*g[2],1.0+0.5*g[0],marker='o',c='brown')
        legend ="g"
        gt = g.reshape(-1,1)
        g38 = 0.5*np.block([-gt,-gt,-gt,-gt,gt,gt,gt,gt])
        x34 = np.empty((3,4))
        shift = np.ones(4)
        for i in range(Lprime.shape[0]):
            for j in range(4):
                x34[:,j] = [0.0,L[neighbours[i,j]][0],L[neighbours[i,j]][1]]
            ax.plot_trisurf(x34[1,:]+g38[1,:4],x34[2,:]+g38[2,:4],x34[0,:]+g38[0,:4],color='brown',facecolor='brown',alpha=0.5)
            ax.plot_trisurf(x34[1,:]+g38[1,4:],x34[2,:]+g38[2,4:],x34[0,:]+g38[0,4:]+shift,color='brown',facecolor='brown',alpha=0.5)
    elif quantity == "A":
        A38 = np.empty((3,8))
        x38 = np.empty((3,8))
        shift = np.ones(4)
        for i in range(Lprime.shape[0]):
            x = Lprime[i]
            A38 = 0.5*np.outer(np.dot(A[:,1:],x),[-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0])
            for j in range(4):
                x38[:,j] = [0.0,L[neighbours[i,j]][0],L[neighbours[i,j]][1]]
                x38[:,j+4] = [1.0,L[neighbours[i,j]][0],L[neighbours[i,j]][1]]
            ax.plot_trisurf(x38[1,:4]+A38[1,:4],x38[2,:4]+A38[2,:4],x38[0,:4]+A38[0,:4],color='cyan',facecolor='cyan',alpha=0.5)
            ax.plot_trisurf(x38[1,4:]+A38[1,4:],x38[2,4:]+A38[2,4:],x38[0,4:]+A38[0,4:],color='cyan',facecolor='cyan',alpha=0.5)
            defPts = ax.scatter(x38[1,:4]+A38[1,:4],x38[2,:4]+A38[2,:4],x38[0,:4]+A38[0,:4],marker='o',c='cyan')
            ax.scatter(x38[1,4:]+A38[1,4:],x38[2,4:]+A38[2,4:],x38[0,4:]+A38[0,4:],marker='o',c='cyan')
        legend ="A-term"
    elif quantity == "thinCorr":
        thinCorr = 0.25*np.dot(A[:,1:],M)#ultrathin correction, $\mathfrak{C}$
        for x in Lprime:
            for i in np.arange(len(z)):
                defPts = ax.scatter(x[0]+z[i,0]+thinCorr[1,i],x[1]+z[i,1]+thinCorr[2,i],0.0+thinCorr[0,i],marker='o',c='pink')
                ax.scatter(x[0]+z[i,0]+thinCorr[1,i+4],x[1]+z[i,1]+thinCorr[2,i+4],1.0+thinCorr[0,i+4],marker='o',c='pink')
        legend="ultrathin corr."
    else:
        for i in range(L.shape[0]):
            x = L[i]
            # defPts = ax.scatter(0.0+alpha[0,i],x[0]+alpha[1,i],x[1]+alpha[2,i],marker='o',c="orange")
            # ax.scatter(1.0+alpha[0,i],x[0]+alpha[1,i],x[1]+alpha[2,i],marker='o',c="orange") #different order of axes
            defPts = ax.scatter(x[0]+alpha[1,i],x[1]+alpha[2,i],0.0+alpha[0,i],marker='o',c="orange")
            ax.scatter(x[0]+alpha[1,i],x[1]+alpha[2,i],1.0+alpha[0,i],marker='o',c="orange")
        legend="alpha"
        #now fill polygons
        alpha34 = np.empty((3,4))
        x34 = np.empty((3,4))
        for i in range(Lprime.shape[0]):
            for j in range(4):
                alpha34[:,j] = alpha[:,neighbours[i,j]]
                x34[:,j] = [0.0,L[neighbours[i,j]][0],L[neighbours[i,j]][1]]
            ax.plot_trisurf(x34[1,:]+alpha34[1,:],x34[2,:]+alpha34[2,:],x34[0,:]+alpha34[0,:],color='orange',facecolor='orange',alpha=0.5)#transform= )#alphaSecondRow,alphaThirdRow,alphaFirstRow #use alpha1 and alpha2 as independent variables and alpha0 as the third one

    ax.set_xbound(-0.5+minY,0.5+maxY)
    ax.set_ybound(-0.5+minZ,0.5+maxZ)
    ax.set_zbound(lower=-0.5,upper=1.5)
    ax.set_xlabel(r"$x_2$")
    ax.set_ylabel(r"$x_3$")
    ax.set_zlabel(r"$x_1$")
    ax.view_init(elev=-120,azim=160,roll=90)
    ax.set_title("")
    ax.legend([refPts,defPts],['reference positions',legend+' with k2={},k3={},t={}'.format(curv2,curv3,torsion)],loc='lower center')
    plt.axis('auto')#square

    plt.show()

plotResult(quantity="alpha")
plotResult(quantity="g")
plotResult(quantity="A")
plotResult(quantity="thinCorr")