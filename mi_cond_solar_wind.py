#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:50:23 2020

@author: mikkosavola
"""
import main_mi as mm
import aux_mi as am
import numpy as np
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
import find_storms as fst
import os

def mi_cond_solar_wind(V_limits,_phase,e_flux_type):  
    #Evaluate conditional mutual information of ULF and eflux, conditioned on
    #V (solar wind speed) for differing levels of V (V below or above the limit)
    
    path="/Users/mikkosavola/Documents/Opiskelu_HY/Gradu/"\
                "raw_data/ViRBO_Space_science/Simms_replication"
    path.replace(" ", "")
    path.replace("\n", "")
    os.chdir(path)
    file1=am.create_path(path,'ViRBO_ULF.txt')
    file2=am.create_path(path,'e_flux.txt')
    
    #Read ULF data to capture solar wind speed V
    ulf,file1=am.reader_VIRBO(file1)
    
    #Empty lists for recording ULF, V and e flux values during storms
    ULF_gr=[]
    ULF_geo=[]
    e_flux=[]
    V=[]
    

    #Parameters for identifying storms
    dst_onset=-30
    dst_rec=-30
    bz_limit=-5
    dst_pre_tt=10
    
    print("Storm parameters: dst_onset=%i, dst_rec=%i, bz_limit=%i, storm duration=%i, e_flux_type=%s" %(dst_onset,dst_rec,bz_limit,dst_pre_tt,e_flux_type))
     
    #Search for storm events and record the corresponding data points
    obj_dst,obj_sgr,obj_sgeo,storms=fst.find_storms(file1,file2,dst_onset,dst_rec,bz_limit,dst_pre_tt,e_flux_type)
    #Number of storms
    n_storms=np.shape(storms)[1]
    #Pick ULF and e flux value from the storms
    for ii in range(n_storms):
        main_phase_start=int(storms[1,ii,1])
        main_phase_end=int(storms[1,ii,2])
        recovery_phase_end=int(storms[1,ii,3])
        _start=main_phase_start
        #Pick correct phases
        if _phase=='main':
            _end=main_phase_end
        elif _phase=='recovery':
            _start=main_phase_end
            _end=recovery_phase_end
        elif _phase=='main+recovery':
            _end=recovery_phase_end  
        ULF_gr.append(obj_sgr.x[_start:_end+1])
        ULF_geo.append(obj_sgeo.x[_start:_end+1])
        e_flux.append(obj_sgeo.y[_start:_end+1])
        V.append(ulf['V'][_start:_end+1])
    #Flatten lists and convert to numpy arrays
    ULF_gr=np.asarray(am.flatten_list(ULF_gr))
    ULF_geo=np.asarray(am.flatten_list(ULF_geo))
    e_flux=np.asarray(am.flatten_list(e_flux))
    V=np.asarray(am.flatten_list(V))
    
    #Convert big values to nans for removal
    _big=9999
    ULF_gr=obj_sgr._big2nans(ULF_gr,_big)
    ULF_geo=obj_sgr._big2nans(ULF_geo,_big)
    V=obj_sgr._big2nans(V,_big)
    e_flux=obj_sgr._big2nans(e_flux,_big)
    
    #Remove nans
    print("Removing nans for ULF_gr and the related e flux and V")
    print("Originally %i data points" %len(e_flux))
    temp_arr=obj_sgeo._remnans_n(ULF_gr,V,e_flux)
    ULF_gr=temp_arr[:,0]
    V_gr=temp_arr[:,1]
    e_flux_gr=temp_arr[:,2]
    print("Removing nans for ULF_geo and the related e flux and V")
    print("Originally %i data points" %len(e_flux))
    temp_arr=obj_sgeo._remnans_n(ULF_geo,V,e_flux)
    ULF_geo=temp_arr[:,0]
    V_geo=temp_arr[:,1]
    e_flux_geo=temp_arr[:,2]
    
    
    #Vectors for storing mutual information and noise for different thresholds of V
    I_gr=np.zeros(len(V_limits))
    I_geo=np.zeros(len(V_limits))
    noise_gr=np.zeros([2,len(V_limits)])
    noise_geo=np.zeros([2,len(V_limits)])
    
    for ii in range(len(V_limits)):
        #print("Evaluation for V=%f" %V_limits[ii])
        print("\nSgr:")
        #I_gr[ii],noise_gr[0,ii],noise_gr[1,ii]=obj_sgr._mi_cond_binary(ULF_gr,e_flux_gr,V_gr,V_limits[ii])
        obj_sgr._mi_cond_binary(ULF_gr,e_flux_gr,V_gr,V_limits[ii])
        #res_gr=[V_limits[ii],I_gr[ii],noise_gr[0,ii],noise_gr[1,ii]]
        #print(res_gr)
        print("\nSgeo:")
        #I_geo[ii],noise_geo[0,ii],noise_geo[1,ii]=obj_sgeo._mi_cond_binary(ULF_geo,e_flux_geo,V_geo,V_limits[ii])     
        obj_sgeo._mi_cond_binary(ULF_geo,e_flux_geo,V_geo,V_limits[ii])     
        #res_geo=[V_limits[ii],I_geo[ii],noise_geo[0,ii],noise_geo[1,ii]]       
        #print(res_geo)
        
    #Plot results
    x_label='Solar wind speed (km/s)'
    y_label="Mutual information (bits)"
    ttl_gr=str("I from "+_phase+" phase btw. ULF ground and e flux "+e_flux_type+", cond. on V")
    ttl_geo=str("I from "+_phase+" phase btw. ULF geo and e flux "+e_flux_type+", cond. on V")
    am.simple_plot(V_limits,I_gr,x_label,y_label,ttl_gr,show=1,save=1,lin_fit=None)
    am.simple_plot(V_limits,I_geo,x_label,y_label,ttl_geo,show=1,save=1,lin_fit=None)  
    
#Go through V_limits
phases=['main','recovery'] #give here 'main', 'recovery' or 'main+recovery'
e_flux_type='Fe1p2'
for ii in phases:
    print("\n\n Analysis for %s phase" %ii)
    V_limits=np.array([250,300,350,400,450,500,550,600,650,700,750,800])
    mi_cond_solar_wind(V_limits,ii,e_flux_type)
    