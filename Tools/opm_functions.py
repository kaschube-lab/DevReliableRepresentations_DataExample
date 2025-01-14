import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def phi_from_map(opm_est):
    phi_est = 0.5*np.angle(opm_est) + np.pi*(np.angle(opm_est)<0)  #2d
    phi_est[phi_est==np.pi] = 0
    return phi_est

def opm_from_tuning(tuning, normkey=True):    
    num_stim, num_cells = tuning.shape
    if num_stim==9:
        tuning = tuning[:8]
        return opm_from_tuning(tuning)
    if num_stim==17:
        tuning = tuning[:16]
        return opm_from_tuning(tuning)
    angles = np.arange(num_stim)*2*np.pi/float(num_stim)
    angles = np.exp(2j*angles)
    #norm=np.nansum(tuning, axis=0)
    norm=np.nansum(np.abs(tuning), axis=0)
    if normkey==False: norm=1
    opm = np.nansum(tuning*angles[:,None], axis=0)/norm
    #/norm
    return opm



def opm_to_rgb_bright(opm, clip):

    def phi_to_hue(phi):
        return phi/np.pi

    def selec_to_S(selec, th):
        #perc=np.percentile(selec[selec>0], th)
        selec=np.clip(selec/th, a_max=1, a_min=0)
        return selec
        #/np.max(selec)
        #np.clip(selec/(selec_max*th), a_max=1, a_min=0)
    
    def phi_from_map(opm_est):
        phi_est = 0.5*np.angle(opm_est) + np.pi*(np.angle(opm_est)<0)  #2d
        phi_est[phi_est==np.pi] = 0
        return phi_est
    
    phi=phi_from_map(opm)
    selec=np.abs(opm)
    H = phi_to_hue(phi)
    S = selec_to_S(selec, clip)
    V = np.ones_like(H)
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def opm_to_rgb(opm, clip):

    def phi_to_hue(phi):
        return phi/np.pi

    def selec_to_S(selec, th):
        #perc=np.percentile(selec[selec>0], th)
        selec=np.clip(selec/th, a_max=1, a_min=0)
        return selec
        #/np.max(selec)
        #np.clip(selec/(selec_max*th), a_max=1, a_min=0)
    
    def phi_from_map(opm_est):
        phi_est = 0.5*np.angle(opm_est) + np.pi*(np.angle(opm_est)<0)  #2d
        phi_est[phi_est==np.pi] = 0
        return phi_est
    
    phi=phi_from_map(opm)
    selec=np.abs(opm)
    H = phi_to_hue(phi)
    V = selec_to_S(selec, clip)
    S = np.ones_like(H)#*0.5
    HSV = np.dstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    return RGB