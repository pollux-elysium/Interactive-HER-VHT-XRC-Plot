import numpy as np
# from MATHEXT import rootFinder
from scipy.optimize import brentq as scipyRootFinder
from .rates import heyrovsky1forward, heyrovsky1reverse, heyrovsky2forward, heyrovsky2reverse, tafelforward, tafelreverse, volmer1forward, volmer1reverse, volmer2forward, volmer2reverse

F = 96485.33212

def rate(params,X,debug=False,thetas=None):
    Y = np.ones(X.shape[1])
    if thetas is None:
        thetas = getThetas(params,X,debug)
    if debug:
        print(f"Thetas: {thetas}")
    intermediate = tafelforward(params,thetas) - tafelreverse(params,thetas) + heyrovsky1forward(params,X[0,:],X[1,:],thetas) - heyrovsky1reverse(params,X[0,:],thetas) + heyrovsky2forward(params,X[0,:],thetas) - heyrovsky2reverse(params,X[0,:],X[1,:],thetas)
    mask = intermediate <0
    Y[~mask] = np.log10(2*1000*F*intermediate[~mask])
    Y[mask] = -100
    return Y


def getThetas(params,X,debug):
    thetas = np.ones(X.shape[1])
    for i in range(X.shape[1]):
        if debug: 
            print(f"Calculating theta for point {i+1} of {X.shape[1]}", end='\r')
        thetas[i] = getTheta(params,X[0,i],X[1,i],debug)
    if debug:
        print("")
    return thetas

def getTheta(params,E,pH,debug):
    f_theta = lambda theta: volmer1forward(params,E,pH,theta) \
        + volmer2forward(params,E,theta) \
        + 2*tafelreverse(params,theta) \
        + heyrovsky1reverse(params,E,theta) \
        + heyrovsky2reverse(params,E,pH,theta) \
        - volmer1reverse(params,E,theta) \
        - volmer2reverse(params,E,pH,theta) \
        - 2*tafelforward(params,theta) \
        - heyrovsky1forward(params,E,pH,theta) \
        - heyrovsky2forward(params,E,theta)
    
    #Debug

    
    try:
        # theta = rootFinder(f_theta,lower=0,upper=1,tol=1e-14,max_iter=100,max_search_expand=0,initial_sampling=2)
        theta = scipyRootFinder(f_theta,0,1)
    except Exception as e:
        # print(f"Warning: Theta not converged for E={E}, pH={pH}, setting to 0")
        # print(params)
        theta = 0.0
    if debug:
        print(f"Volmer 1 Forward: {volmer1forward(params,E,pH,theta)}")
        print(f"Volmer 1 Reverse: {volmer1reverse(params,E,theta)}")
        print(f"Volmer 2 Forward: {volmer2forward(params,E,theta)}")
        print(f"Volmer 2 Reverse: {volmer2reverse(params,E,pH,theta)}")
        print(f"Tafel Forward: {tafelforward(params,theta)}")
        print(f"Tafel Reverse: {tafelreverse(params,theta)}")
        print(f"Heyrovsky 1 Forward: {heyrovsky1forward(params,E,pH,theta)}")
        print(f"Heyrovsky 1 Reverse: {heyrovsky1reverse(params,E,theta)}")
        print(f"Heyrovsky 2 Forward: {heyrovsky2forward(params,E,theta)}")
        print(f"Heyrovsky 2 Reverse: {heyrovsky2reverse(params,E,pH,theta)}")


    return theta

def getZ(params,Es,pHs,thetas=None):
    if thetas is None:
        thetas = [getTheta(params,E,pH,debug=False) for E,pH in zip(Es,pHs)]
    Z = np.zeros((len(Es),5))
    for i,(E,pH,theta) in enumerate(zip(Es,pHs,thetas)):
        # Z[i,0] = volmer1forward(params,E,pH,theta)/volmer1reverse(params,E,theta)
        Z[i,0] = volmer1reverse(params,E,theta)/volmer1forward(params,E,pH,theta)
        Z[i,1] = volmer2reverse(params,E,pH,theta)/volmer2forward(params,E,theta)
        Z[i,2] = tafelreverse(params,theta)/tafelforward(params,theta)
        Z[i,3] = heyrovsky1reverse(params,E,theta)/heyrovsky1forward(params,E,pH,theta)
        Z[i,4] = heyrovsky2reverse(params,E,pH,theta)/heyrovsky2forward(params,E,theta)
    return Z

def getXrc(params,Es,pHs,dlogk=0.001):
    
    dlogk1 = dlogk*params[2]
    dlogk2 = dlogk*params[5]
    dlogkT = dlogk*params[8]
    dlogk4 = dlogk*params[10]
    dlogk5 = dlogk*params[13]

    params1n = params.copy()
    params1n[2] -= dlogk1
    params1p = params.copy()
    params1p[2] += dlogk1
    params2n = params.copy()
    params2n[5] -= dlogk2
    params2p = params.copy()
    params2p[5] += dlogk2
    paramsTn = params.copy()
    paramsTn[8] -= dlogkT
    paramsTp = params.copy()
    paramsTp[8] += dlogkT
    params4n = params.copy()
    params4n[10] -= dlogk4
    params4p = params.copy()
    params4p[10] += dlogk4
    params5n = params.copy()
    params5n[13] -= dlogk5
    params5p = params.copy()
    params5p[13] += dlogk5

    Xrc1 = (rate(params1p,np.array([Es,pHs])) - rate(params1n,np.array([Es,pHs])))/(2*dlogk1)
    Xrc2 = (rate(params2p,np.array([Es,pHs])) - rate(params2n,np.array([Es,pHs])))/(2*dlogk2)
    XrcT = (rate(paramsTp,np.array([Es,pHs])) - rate(paramsTn,np.array([Es,pHs])))/(2*dlogkT)
    Xrc4 = (rate(params4p,np.array([Es,pHs])) - rate(params4n,np.array([Es,pHs])))/(2*dlogk4)
    Xrc5 = (rate(params5p,np.array([Es,pHs])) - rate(params5n,np.array([Es,pHs])))/(2*dlogk5)

    Xrc = np.array([Xrc1,Xrc2,XrcT,Xrc4,Xrc5]).T
    return Xrc

def finiteDiffAlpha(params,Es,pHs,dE = 0.0001):
    djde = (rate(params,np.array([Es+dE,pHs])) - rate(params,np.array([Es-dE,pHs])))/(2*dE)
    tafelSlope = 1000/djde
    alpha = -60/tafelSlope
    return alpha

def finiteDiffRho(params,Es,pHs,dpH = 0.001):
    djdpH = (rate(params,np.array([Es,pHs+dpH])) - rate(params,np.array([Es,pHs-dpH])))/(2*dpH)
    rho = -djdpH
    return rho