# -*- coding: utf-8 -*-

def simulateurTPI(D0=2e-2,L=1,phi=0.0005,lambda0=785e-9,f=5e-2,n=31,deltazMin=-10e-6,deltazMax=10e-6,xMax=1.2e-2,xMin=-1.2e-2,yMax=1.2e-2,yMin=-1.2e-2,pixelsx=200,pixelsy=200):
    """
    
    Function that reproduce the images of the TPI interferometer.
    
    References: 
        https://www.researchgate.net/publication/337679211_Submicrometric_Absolute_Positioning_of_Flat_Reflective_Surfaces_Using_Michelson_Interferometry
    
    In:
        D0: incident beam diameter
        L: Distance between observation plane & the reference mirror / lens [m]
        phi: angle between incident and reflected beams on the reference mirror [rad]
        lambda0: laser wavelength [m]
        f: lens focal length [m]
        pixelsx: camera x-pixels number
        pixelsy: camera y-pixels number
        deltazMin: minimum target position along focal spot, relative to f2 position [m]
        deltazMax: maximum target position along focal spot, relative to f2 position [m]
        n: number of images to generate along deltaz
        xMax: maximum x-position of the observation screen [m]
        xMin: minimum x-position of the observation screen [m]
        yMax: maximum y-position of the observation screen [m]
        yMin: minimum y-position of the observation screen [m]
        
    Out:
        
    """
    
#Libraries
    import numpy as np
    from matplotlib import image as img
    
#Verification of function entry parameters
    
#Initialization of vectors and constant
    
    deltaz = np.linspace(deltazMin,deltazMax,n)
    x = np.linspace(xMin,xMax,pixelsx);
    y = np.linspace(yMin,yMax,pixelsy);
    
    k0 = 2*np.pi/lambda0
    
#Computation of gaussian beam 1 parameters
    
    zf = -f * ((D0/4)**2 / ((D0/4)**2 + (lambda0*f/np.pi)**2)) #Distance focal point - lens
    
    invq2_zA = complex(-1/f,-4*lambda0/(np.pi*D0**2))
    invq2_zB = invq2_zA/(1-2*(zf-deltaz)*invq2_zA)
    invq3_zB = invq2_zB - 1/f
    invq3_zL = invq3_zB/(1+L*invq3_zB)
    
    w1 = np.sqrt(-lambda0/(np.pi*invq3_zL.imag))
    R1 = 1/invq3_zL.real
    z1 = w1**2*R1/(w1**2+(lambda0*R1/np.pi)**2)
    w01 = np.sqrt(abs(z1*lambda0/np.pi * np.sqrt(R1/z1-1)))
    gp1 = np.arctan(z1*lambda0/(np.pi*w01**2));
    
    for i in np.arange(n):
        if w01[i] > w1[i]:
            w01[i] = w1[i]
    
#Computation of gaussian beam 2 parameters
    
    w2 = D0/2
    R2 = 1e20
    w02 = D0/2
    
#Computation of intensity for each pixels
    
    I = np.zeros((pixelsx,pixelsy,n))
    
    for i in np.arange(pixelsx):
        for j in np.arange(pixelsy):
            for k in np.arange(n):
                #Calcul des r
                L2 = L*np.cos(phi) + y[j]*np.sin(phi)
                y2 = L2*np.sin(phi)
                z2 = L2*np.cos(phi) - L
                gp2 = np.arctan(z2*lambda0/(np.pi*w02**2));
                
                r1square = x[i]**2 + y[j]**2
                r2square = x[i]**2 + (y[j]-y2)**2 + z2**2
                
                I[i,j,k] = (w01[k]/w1[k])**2 * np.exp(-2*r1square/(w1[k]**2)) + (w02/w2)**2 * np.exp(-2*r2square/(w2**2)) + 2 * w01[k]/w1[k] * w02/w2 * np.exp(-1*r1square/(w1[k]**2)) *  np.exp(-1*r2square/(w2**2)) * np.cos(phi) * np.cos(k0*(z2-z1[k]) + k0/2 * (r2square/R2 - r1square/R1[k]) - (gp2 - gp1[k]))
    
    destinationFolder = 'imagesTPI/';
    
    for i in np.arange(n):
        img.imsave(destinationFolder+'img{0}.png'.format(i), np.transpose(I[:,:,i]), cmap='gray')
 
    return w01,w1