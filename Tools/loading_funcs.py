import numpy as np

import sys, os
from os.path import abspath, sep, pardir
sys.path.append(abspath('') + sep + pardir)
#create basepath based on folder containing 'DataF2181', 'Analysis', and others
basepath=abspath('') + sep + pardir

def load_grating_evoked_data(key='binocular',
                            all_dates=['-2','+0','+2','+4','+6'],
                            basepath=basepath,
                            animal='2181'):
    
    #dict
    grating_frames_all = {}
    
    #loop trough dates 
    for date in all_dates:
        all_tp_data_day = np.load(
            "{}/DataF{}/Day_EO{}/evoked_data_{}.npy".format(basepath,animal, date, key))
                
            
        list_tp=[1,8,16,24,31,39,46,54]
        label_tp=['0s', '0.5s', '1s', '1.5s', 
                     '2s', '2.5s', '3s','3.5s',]
        grating_frames_day={}
        for tp in range(len(list_tp)):
            grating_frames_day[label_tp[tp]]= all_tp_data_day[tp]
        del all_tp_data_day     
                
        grating_frames_all.update({'{}'.format(date) : grating_frames_day})        
    return grating_frames_all


def load_spont_data(all_dates=['-2','+0','+2','+4','+6'],
                    basepath=basepath,
                    animal='2181'):

    
    #dict
    spont_frames_all = {}
    #loop trough dates 
    for date in all_dates:
        spont_frames_day = np.load(
            "{}/DataF{}/Day_EO{}/spont_data.npy".format(basepath,animal, date))
        
        spont_frames_all.update({'{}'.format(date) : spont_frames_day})
              
    #return
    return spont_frames_all

