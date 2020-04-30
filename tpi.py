# -*- coding: utf-8 -*-

#Decorator to measure execution time
def measureExecutionTime(fct):
    def newfct(*p1,**p2):
        import time
        ti = time.time()
        fct(*p1,**p2)
        tf = time.time()
        return tf-ti
    return newfct

#Simulation of TPI images
def simulateurTPI(D0=2e-2,L=1,phi=0.0005,lambda0=785e-9,f=5e-2,n=31,deltazMin=-20e-6,deltazMax=20e-6,xMax=0.6e-2,xMin=-0.6e-2,yMax=0.6e-2,yMin=-0.6e-2,pixelsx=200,pixelsy=200,destinationFolder = 'imagesTPI'):
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
        destinationFolder: location and name of the directory for the images
        
    Out:
        
    """
    
#Libraries
    import numpy as np
    from matplotlib import image as img
    import os
    
#Initialization of vectors and constant
    
    deltaz = np.linspace(deltazMin,deltazMax,n)
    x = np.linspace(xMin,xMax,pixelsx);
    y = np.linspace(yMin,yMax,pixelsy);
    
    k0 = 2*np.pi/lambda0
    
#Computation of gaussian beam 1 parameters
    
    #Distance focal point - lens
    zf = -f * ((D0/4)**2 / ((D0/4)**2 + (lambda0*f/np.pi)**2))
    
    #Gaussian beam propagation
    invq2_zA = complex(-1/f,-4*lambda0/(np.pi*D0**2))
    invq2_zB = invq2_zA/(1-2*(zf-deltaz)*invq2_zA)
    invq3_zB = invq2_zB - 1/f
    invq3_zL = invq3_zB/(1+L*invq3_zB)
    
    #Gaussian beam parameters at the observation plane
    w1 = np.sqrt(-lambda0/(np.pi*invq3_zL.imag))
    R1 = 1/invq3_zL.real
    z1 = w1**2*R1/(w1**2+(lambda0*R1/np.pi)**2)
    w01 = np.sqrt(abs(z1*lambda0/np.pi * np.sqrt(R1/z1-1)))
    gp1 = np.arctan(z1*lambda0/(np.pi*w01**2));
    
    #Needed correction for highly colimated beam
    #for i in np.arange(n):
     #   w01[i] = w1[i]
    
#Computation of gaussian beam 2 parameters
    
    w2 = D0/2
    R2 = 1e20
    w02 = D0/2
    
#Generation of simulated images
    
    #Initialization images
    I = np.zeros((pixelsx,pixelsy,n))
    
    #Compute intensity on each pixel of each image
    for i in np.arange(pixelsx):
        for j in np.arange(pixelsy):
            for k in np.arange(n):
                                
                L2 = L*np.cos(phi) + y[j]*np.sin(phi)
                y2 = L2*np.sin(phi)
                z2 = L2*np.cos(phi) - L
                gp2 = np.arctan(z2*lambda0/(np.pi*w02**2));
                r1square = x[i]**2 + y[j]**2
                r2square = x[i]**2 + (y[j]-y2)**2 + z2**2
                
                I[i,j,k] = (w02/w1[k])**2 * np.exp(-2*r1square/(w1[k]**2)) + (w02/w2)**2 * np.exp(-2*r2square/(w2**2)) + 2 * w02/w1[k] * w02/w2 * np.exp(-1*r1square/(w1[k]**2)) *  np.exp(-1*r2square/(w2**2)) * np.cos(phi) * np.cos(k0*(z2-z1[k]) + k0/2 * (r2square/R2 - r1square/R1[k]) - (gp2 - gp1[k]))
    
    #Creation or verification of existance of the destination folder
    if not os.path.isdir(destinationFolder):
        os.mkdir(destinationFolder)
    
    #Save images
    for i in np.arange(n):
        img.imsave(destinationFolder+'/img{0}.png'.format(i), np.transpose(I[:,:,i]), cmap='gray')
        

#Compute ODF
def functionODF(img,step=1):
    
    """
        
        Function that compute the orientation density function (ODF) of an image.
        
        In:
            img: Image to process (.png)
            step: Angle step for Radon transform [degrees]
        Out: 
            theta: vector of angles [degrees]
            odf: vector ODF values
        
    """
        
#Libraries
    
    from PIL import Image
    import numpy as np
    from skimage.transform import radon
    
#Verify if image is square and resize if not

    (Lx,Ly) = img.size
    L = max(Lx,Ly)
    
    if Lx != Ly:
        img = img.resize((L,L),Image.BICUBIC)
    
#Transform image into matrix and apply circular mask
    
    x0 = y0 = (L+1)/2
    rmax = L/2
    
    img = np.asarray(img)
    img2d = np.zeros((L,L))
        
    for i in np.arange(L):
        for j in np.arange(L):
            r = np.sqrt((i-x0)**2+(j-y0)**2)
            if r <= rmax:
                img2d[i,j] = img[j][i][0]
    
#Apply Radon and Fourier transform 
            
    theta = np.arange(0,180,step)
    imgRad = radon(img2d,theta,circle=False)
    imgFT = abs(np.fft.fftshift(np.fft.fft(imgRad,axis=0)))

#Compute odf

    odf = np.sum(imgFT,0)
    odf = odf/sum(odf)
    
#Output parameters
    return odf,theta


#Measure the mean angle of the fringes on image
def measureAngleInterferenceFringe(img,step):
    
    """
        
        Function that measure the mean orientation of an image fringes.
        
        In:
            img: Image to process (.png)
            step: Angle step for Radon transform [degrees]
        Out: 
            meanTheta: mean fringes angle [degrees]
            rsquared: R^2
        
    """
    
    from scipy.optimize import curve_fit
    import numpy as np
    
    odf,theta = functionODF(img,step)
    
    def gauss(x,A,mu,sigma,d):
        return A*np.exp(-(x-mu)**2/(2*sigma**2)) + d
    
    def gauss_fit(x, y):
        mu = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mu) ** 2) / sum(y))
        popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mu, sigma])
        return popt
    
    p = gauss_fit(theta, odf)
    
    res = odf - gauss(theta,*p)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((odf-np.mean(odf))**2)
    rsquared = 1 - (ss_res/ss_tot)

    return p[1],rsquared