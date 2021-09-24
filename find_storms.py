#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:01:38 2020

@author: mikkosavola
"""
def find_storms(file1,file2,dst_onset,dst_rec,bz_limit,dst_pre_tt,e_index,plott=None):
    import aux_mi as am
    import main_mi as mm
    import numpy as np
    #import scipy.stats as stats
    import os
    import matplotlib.pyplot as plt
    
    #Read data from input files
    ulf,file1=am.reader_VIRBO(file1)
    eflux,file2=am.reader_EFLUX(file2)
    dates=[eflux['year'],eflux['doy'],eflux['hour']]
    Lshells=1
     
    #Create objects containing Dst, ULF spectral power, Bz and e flulx
    obj_dst=mm.My_Mi(file1,dates,ulf['Dst'],file2,Lshells,ulf['Bz'])
    obj_sgr=mm.My_Mi(file1,dates,ulf['Sgr'],file2,Lshells,eflux[e_index])
    obj_sgeo=mm.My_Mi(file1,dates,ulf['Sgeo'],file2,Lshells,eflux[e_index])
    
    
    #Find the beginning of each storm by searching in chunks of n data points
    #and omitting a minimum if it occurs within the last k points of a chunk
    #Store the resulting indices in dst_mins
    dst_onset=dst_onset
    dst_rec=dst_rec
    bz_limit=bz_limit
    n=40 #orig. 20
    k=int(19) #orig. 5
    
    #Time stamps of storm phases in hours
    dur_pre=12
    dur_after=48
    dur_recovery=24
    
    #Find the monotonically decreasing sequences in Dst
    decr_dst=am.decreasing_sequences(obj_dst.x)
    
    #Set the segment length for searching for storms
    l_dst_x=len(obj_dst.x)
    segments=int(np.ceil(l_dst_x/n))
    #duration=[]
    
    
    #Record in the array below the Dst values for the beginning of pre phase,
    #Dst maximum, Dst minimum and the end of recovery phase: x axis contains place
    #for index value (x[0]) and indicdes (x[1], y contains "labels" for pre phase
    #Dst maximum, Dst minimum and recovery phase
    #z axis contains the values and the indices (i.e. running time stamps)
    #storms=np.zeros([2,4,1])
    dst_minima=np.zeros([2,0])
    dst_maxima=np.zeros([2,0])
    dst_recovery=np.zeros([2,0])
    ind_rec=-(dur_after+dur_recovery)
    
    
    ######FIND DST MAXIMA AND MINIMA AND THEIR INDICES IN THE ULF RECORD
    for ii in range(segments-1):
        #Look for minimum Dst
        #Address the final segment separately, in case its length is less than n
        if ii==segments-2:
            ind_min=am.find_min_dst(obj_dst.x[ii*n:l_dst_x],n,k,dst_onset)
        else:
            ind_min=am.find_min_dst(obj_dst.x[ii*n:(ii+2)*n],n,k,dst_onset)
        if ind_min is False:
            continue
        else: 
            #Find Dst minimum and record it, also record it as ind_rec for
            #comparison in the next iteration round
            ind_min=ind_min+ii*n      
            #Dst at the beginning of main phase must be higher than the threshold
            #of dst_onset nT for a minimum
            dst=obj_dst.x
            ind_begin=am.find_main_phase(decr_dst,ind_min,0,dst,dst_onset)
            #Check that at least 72 h have elapsed since the end
            #of the recovery phase
            if ind_begin-ind_rec<72:
                continue
            #Check that there's no smaller Dst value between the maximum and
            #the minimum
            if not np.all(obj_dst.x[ind_begin:ind_min]>obj_dst.x[ind_min]):
                continue
            #Check that Bz minimum is below -5 nT during the main phase
            bz=obj_dst.y  
            if am.check_min_bz(bz[ind_begin:ind_min+1],bz_limit):    
                #Find the next occurrence of Dst >-30 nT after the Dst minimum
                #and mark that as ind_rec, which here denotes the end of
                #recovery phase
                ind_rec=am.find_recovery(obj_dst.x[ind_min:],dst_rec)
                ind_rec=ind_rec+ind_min
                #Check that storm duration is at least dst_pre_tt hours
                if ind_rec-ind_begin<dst_pre_tt:
                    continue
                #Check that recovery is fast enough
                if ind_rec-ind_min>dur_after+dur_recovery:
                    continue
                #Save the minimum Dst and maximum Dst and the respective indices
                arr1=np.array([ulf['Dst'][ind_begin],ind_begin],dtype=np.float64,ndmin=2).T
                arr2=np.array([ulf['Dst'][ind_min],ind_min],dtype=np.float64,ndmin=2).T
                dst_maxima=np.concatenate((dst_maxima,arr1),axis=1)
                dst_minima=np.concatenate((dst_minima,arr2),axis=1)
                #Save end of storm to an array
                arr3=np.array([ulf['Dst'][ind_rec],ind_rec],dtype=np.float64,ndmin=2).T
                dst_recovery=np.concatenate((dst_recovery,arr3),axis=1)
                    
    
    #Create an array containing the storms
    storms=np.empty([2,dst_maxima.shape[1],4])
    #Indices of the Dst maxima
    max_ind=np.array(dst_maxima[1,:],dtype=np.int64)
    #pre-storm Dst and its index number
    storms[:,:,0]=np.array([obj_dst.x[max_ind-dur_pre],max_ind-dur_pre])
    #Dst maxima, i.e. storm beginning Dst and its index number
    storms[:,:,1]=dst_maxima
    #Dst minima and its index number
    storms[:,:,2]=dst_minima
    #End of recovery phase Dst and its index number
    storms[:,:,3]=dst_recovery
    #The above array contains the Dst values and the respective index values
    #for pre storm, Dst drop, Dst minimum
    #and end of recovery phase.
    
    
    coordinates=zip(storms[1,:,0],storms[1,:,1],storms[1,:,2],storms[1,:,3])
    coordinates=list(coordinates)
    
    #Plot storms
    if plott is not None:
        for ii in coordinates:
            if max(ii)>0:
                if max(ii)<1000000:
                    ii=np.int64(ii)
                    a=ii[0]
                    b=ii[3]
                    x_coord=np.linspace(a,b,b-a+1,dtype=np.int64)
                    plt.plot(x_coord,obj_dst.x[x_coord],label="Dst")
                    plt.plot(x_coord,obj_dst.y[x_coord],label="Bz")
                    plt.scatter(ii[0],obj_dst.x[ii[0]],label="pre storm begin")
                    plt.scatter(ii[1],obj_dst.x[ii[1]],label="Dst max")
                    plt.scatter(ii[2],obj_dst.x[ii[2]],label="Dst min")
                    plt.scatter(ii[3],obj_dst.x[ii[3]],label="end of recovery")
                    plt.xlabel("hourly index")
                    plt.ylabel("nT")
                    plt.legend()
                    plt.show()
    return obj_dst,obj_sgr,obj_sgeo,storms
