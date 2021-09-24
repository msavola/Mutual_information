#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:50:19 2020

@author: mikkosavola
"""
import aux_mi as am
import main_mi as mm
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import find_storms as fst

def simms_replication_averages(e_flux_type):
    #This function replicates the Simms study by evaluating the correlation
    #Between the average ULF spectral power and the maximum of e flux,
    #as described in Simms et al. 2014
    #The function also evaluates the mutual information for the same
    #averaged data sets

    path="/Users/mikkosavola/Documents/Opiskelu_HY/Gradu/"\
                "raw_data/ViRBO_Space_science/Simms_replication"
    path.replace(" ", "")
    path.replace("\n", "")
    os.chdir(path)
    file1=am.create_path(path,'ViRBO_ULF.txt')
    file2=am.create_path(path,'e_flux.txt')
    

    #Parameters for identifying storms
    dst_onset=-30
    dst_rec=-30
    bz_limit=-5
    dst_pre_tt=10
     
    #Search for storm events and reconrd the correesponding data points
    obj_dst,obj_sgr,obj_sgeo,storms=fst.find_storms(file1,file2,dst_onset,dst_rec,bz_limit,dst_pre_tt,e_flux_type)
    ULF_gr=obj_sgr.x
    ULF_geo=obj_sgeo.x
    e_flux=obj_sgeo.y
    
    #Number of storms
    n_storms=np.shape(storms)[1]
    
    #Arrays to store the averages per storm and phase
    avg_ULF_gr=np.empty([3,n_storms])
    avg_ULF_geo=np.empty([3,n_storms])
    #Arrays to store correlations and p values
    corr_ULF_gr=np.empty([3,2])
    corr_ULF_geo=np.empty([3,2])
    #Arrays to store mutual information and its noise per phase
    mi_ULF_gr=np.empty([3,5])
    mi_ULF_geo=np.empty([3,5])
    
    #Array to store e flux maxima 48-78 hours after Dst minimum
    e_flux_max=np.empty(n_storms)
    
    #Evaluate ULF spectral power averages for the storm periods and e flux max
    #during 48-72 hours after Dst minimum
    for ii in range(n_storms):
        pre_phase_start=int(storms[1,ii,0])
        main_phase_start=int(storms[1,ii,1])
        main_phase_end=int(storms[1,ii,2])
        recovery_phase_end=int(storms[1,ii,3])
        

        avg_ULF_gr[0,ii]=np.nanmean(ULF_gr[pre_phase_start:main_phase_start+1])
        avg_ULF_geo[0,ii]=np.nanmean(ULF_geo[pre_phase_start:main_phase_start+1])
        avg_ULF_gr[1,ii]=np.nanmean(ULF_gr[main_phase_start:main_phase_end+1])
        avg_ULF_geo[1,ii]=np.nanmean(ULF_geo[main_phase_start:main_phase_end+1])
        avg_ULF_gr[2,ii]=np.nanmean(ULF_gr[main_phase_end:main_phase_end+48+1])
        avg_ULF_geo[2,ii]=np.nanmean(ULF_geo[main_phase_end:main_phase_end+48+1])
                                          
        e_flux_max[ii]=np.max(e_flux[main_phase_end+48:main_phase_end+72+1])
        
    #Remove nans and "big" values
    big=9999
    e_flux_max=obj_sgr._big2nans(e_flux_max,big) 
    
    phases=["pre","main","recovery"]
    
    #Calculate correlations and mi for ULF geo
    for jj in range(np.shape(corr_ULF_geo)[0]):
        #Remove nans from ULF geo and create the corresponding e flux max array
        print("\nRemoving nans for ULF geo %s" %phases[jj])
        avg_ULF_geo_temp,e_flux_max_geo=obj_sgeo._remnans(avg_ULF_geo[jj,:],e_flux_max,prnt=1)     
        e_flux_max_geo, avg_ULF_geo_temp=obj_sgeo._remnans(e_flux_max_geo,avg_ULF_geo_temp,prnt=1)
        #evaluate  and mi
        corr_ULF_geo[jj,:]=stats.pearsonr(avg_ULF_geo_temp,e_flux_max_geo)
        temp_mi=np.asarray(obj_dst._mi_w_noise(avg_ULF_geo_temp,e_flux_max_geo))
        temp_mi[4]=temp_mi[4][0]
        mi_ULF_geo[jj,:]=temp_mi
        if abs(e_flux_max_geo.all())>9999:
            print("E flux max too large")
    #Calculate correlations and mi for ULF ground
    for jj in range(np.shape(corr_ULF_gr)[0]):
        #Remove nans from ULF gr and create the corresponding e flux max array
        print("\nRemoving nans for ULF gr %s" %phases[jj])
        avg_ULF_gr_temp,e_flux_max_gr=obj_sgr._remnans(avg_ULF_gr[jj,:],e_flux_max,prnt=1)
        e_flux_max_gr,avg_ULF_gr_temp=obj_sgr._remnans(e_flux_max_gr,avg_ULF_gr_temp,prnt=1)
        #Evaluate correlations and mi
        corr_ULF_gr[jj,:]=stats.pearsonr(avg_ULF_gr_temp,e_flux_max_gr)
        temp_mi=np.asarray(obj_dst._mi_w_noise(avg_ULF_gr_temp,e_flux_max_gr))
        temp_mi[4]=temp_mi[4][0]
        mi_ULF_gr[jj,:]=temp_mi
        if abs(e_flux_max_gr.all())>9999:
            print("E flux max too large")
            
        
    #Print results
    print("\nCorrelation and MI of ULF ground measurements and e flux")
    print("12 h pre storm: corr = %f4, p = %f4" %(corr_ULF_gr[0,0],corr_ULF_gr[0,1]))
    print("main phase: corr = %f4, p = %f5" %(corr_ULF_gr[1,0],corr_ULF_gr[1,1]))
    print("Up to 48 h after Dst minimum: corr = %f4, p = %f5 \n" %(corr_ULF_gr[2,0],corr_ULF_gr[2,1]))
    print("12 h pre storm: MI = %f4, noise = %f4, sigma = %4f" %(mi_ULF_gr[0,0],mi_ULF_gr[0,1],mi_ULF_gr[0,2]))
    print("main phase: MI = %f4, noise = %f4, sigma = %4f" %(mi_ULF_gr[1,0],mi_ULF_gr[1,1],mi_ULF_gr[1,2]))
    print("Up to 48 h after Dst minimum: MI = %f4, noise = %f4, sigma = %4f \n" %(mi_ULF_gr[2,0],mi_ULF_gr[2,1],mi_ULF_geo[2,2]))
    
    print("\nCorrelation and MI of ULF geostat. measurements and e flux")
    print("12 h pre storm: corr = %f4, p = %f4" %(corr_ULF_geo[0,0],corr_ULF_geo[0,1]))
    print("main phase: corr = %f4, p = %f4" %(corr_ULF_geo[1,0],corr_ULF_geo[1,1]))
    print("Up to 48 h after Dst minimum: corr = %f4, p = %f4 \n" %(corr_ULF_geo[2,0],corr_ULF_geo[2,1]))
    print("12 h pre storm: MI = %f4, noise = %f4, sigma = %4f" %(mi_ULF_geo[0,0],mi_ULF_geo[0,1],mi_ULF_geo[0,2]))
    print("main phase: MI = %f4, noise = %f4, sigma = %4f" %(mi_ULF_geo[1,0],mi_ULF_geo[1,1],mi_ULF_geo[1,2]))
    print("Up to 48 h after Dst minimum: MI = %f4, noise = %f4, sigma = %4f \n" %(mi_ULF_geo[2,0],mi_ULF_geo[2,1],mi_ULF_geo[2,2]))                             
    
#Run with Fe130 for seed electrson and Fe1p2 for relativistic electrons   
simms_replication_averages('Fe1p2')
    