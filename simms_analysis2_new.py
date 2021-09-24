#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:08:10 2020

@author: mikkosavola
"""
import aux_mi as am
import main_mi as mm
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import find_storms as fst
import time

def simms_analysis_2(storm_param,storm_data,e_flux_type,t_offset,draw=None,prnt_result=None,ret_result=None):
    
    #storm_param is used to give the criteria for storms: Dst limits for storm
    #onset tnad recovery phase's end, thereshold for B_z during a storm's main
    #phase and a mninimum duration for a storm, tt.
    
    #Calculate mutual information and its erros using a shuffle test for
    #ULF Sgr and Sgeo for e flux of type e_flux_tyoe
    #Also returns the results in 4x3 a numpy array, if ret_result is not None
    
    #Positive time offset delays the e flux  with respect to the ULF spectral
    #power, which is taken from the actual storms
    
    #If prnt_result is not None, the numerical results are printed in text
    #If draw is no None, the function plot the values of e flux and ULG spectral
    #power and also does a linear fit and calculates the correaltion.

    
    
    #Parameter for plotting Sgeo and Sgr hourly values per storm and phase:
    #if set to None, no plotting is done
    pltt=None
    
    #ULF fluxes and e flux without division into storm events
    dst=storm_data[0].x
    sgr=storm_data[1].x
    sgeo=storm_data[2].x
    eflux=storm_data[2].y
    storms=storm_data[3]
    #dates=np.linspace(1,len(eflux),len(eflux),dtype=np.int64)
    
    #Truncate storms so that t_offset doesn't go out of bounds
    new_storms=storms
    while new_storms[1,-1,3]+t_offset>storms[1,-1,3]:
        new_storms=new_storms[:,0:-2,:]
    storms=new_storms
    
    #Vectors for saving storm time ULF and e flux
    eflux_sgeo_main=[]
    eflux_sgr_main=[]
    eflux_sgeo_rec=[]
    eflux_sgr_rec=[]
    sgeo_main=[]
    sgeo_rec=[]
    sgr_main=[]
    sgr_rec=[]
    
    #Record the number of points in total in the respective phases over all storms
    n_sgeo_main=0
    n_sgr_main=0
    n_sgeo_rec=0
    n_sgr_rec=0
    print("# of storms:"+str(len(storms[1,:,1])))
    
    #For checking that Sgeo and Sgr contain no consecutive values
    # for kk in range(len(sgeo)-2):
    #     if sgeo[kk+1]==sgeo[kk] and sgeo[kk+2]==sgeo[kk]:
    #         print("Consecutive values at Sgeo at "+str(kk))
    
    # for kk in range(len(sgr)-2):
    #     if sgr[kk+1]==sgeo[kk] and sgr[kk+2]==sgr[kk]:
    #         print("Consecutive values at Sgr at "+str(kk))
    
    #N_gr_main=0 #For recording the number of evaluated data points
    #N_geo_main=0
    #N_gr_rec=0
    #N_geo_rec=0
    
    #Find storm events and the corresponding ULF and e flux values
    for ii in range(len(storms[1,:,1])):
        #ii=np.int64(ii)
        max_ind=storms[1,ii,1].astype(int)
        min_ind=storms[1,ii,2].astype(int)
        rec_ind=storms[1,ii,3].astype(int)
        main_phase=np.linspace(max_ind,min_ind,min_ind-max_ind+1,dtype=np.int64)
        rec_phase=np.linspace(min_ind,rec_ind,rec_ind-min_ind+1,dtype=np.int64)
        #Transform the "too big" values in e flux to Nans
        big=9999
        #Here the e flux is postponed by t_offset
        eflux_main=obj_sgeo._big2nans(eflux[main_phase+t_offset],big)
        eflux_rec=obj_sgeo._big2nans(eflux[rec_phase+t_offset],big)
        if abs(max(eflux_main)>9999):
            print("e flux is too big")
        if abs(max(eflux_rec)>9999):
            print("e flux is too big")
        #Check whether all values in any vector are Nans, if so, MI is not evaluated.
        nan_check1=am.all_nans(sgeo[main_phase]) or am.all_nans(eflux_main)
        nan_check2=am.all_nans(sgr[main_phase]) or am.all_nans(eflux_main)
        nan_check3=am.all_nans(sgeo[rec_phase]) or am.all_nans(eflux_rec)
        nan_check4=am.all_nans(sgr[rec_phase]) or am.all_nans(eflux_rec)
        #Check that e flux data contains no more than n_recurr recurring values at any location
        n_recurr=3
        #Calculate mutual info of events for main phase                                                
        if am.recurring_values(eflux_main,n_recurr)==False:    
            if nan_check1 is False:
                sgeo_temp,eflux_temp=obj_sgeo._tremnans(sgeo[main_phase],eflux_main,0)[0:2]
                sgeo_main.append(sgeo_temp)
                eflux_sgeo_main.append(eflux_temp)
                n_sgeo_main=n_sgeo_main+len(sgeo_temp)
                #N_geo_main=N_geo_main+len(main_phase)
                #Plot Dst
                if pltt is not None:
                    am.simple_plot(main_phase,sgeo[main_phase],"Hourly index","Sgeo","Main Phase Sgeo",show=1)
            
            if nan_check2 is False:
                sgr_temp,eflux_temp=obj_sgr._tremnans(sgr[main_phase],eflux_main,0)[0:2]
                sgr_main.append(sgr_temp)
                eflux_sgr_main.append(eflux_temp)
                n_sgr_main=n_sgr_main+len(sgr_temp)
                #N_gr_main=N_gr_main+len(main_phase)
                #Plot Dst
                if pltt is not None:
                    am.simple_plot(main_phase,sgr[main_phase],"Hourly index","Sgr","Main Phase Sgr",show=1)
        
        #Calculate mutual info of events for recovery phase   
        if am.recurring_values(eflux_rec,n_recurr)==False:    
            if nan_check3 is False:
                sgeo_temp,eflux_temp=obj_sgeo._tremnans(sgeo[rec_phase],eflux_rec,0)[0:2]
                sgeo_rec.append(sgeo_temp)
                eflux_sgeo_rec.append(eflux_temp)
                n_sgeo_rec=n_sgeo_rec+len(sgeo_temp)
                #N_geo_rec=N_geo_rec+len(rec_phase)
                #Plot Dst
                if pltt is not None:
                    am.simple_plot(rec_phase,sgeo[rec_phase],"Hourly index","Sgeo","Recovery Phase Sgeo",show=1)
                
            if nan_check4 is False:
                sgr_temp,eflux_temp=obj_sgr._tremnans(sgr[rec_phase],eflux_rec,0)[0:2]
                sgr_rec.append(sgr_temp)
                eflux_sgr_rec.append(eflux_temp)
                n_sgr_rec=n_sgr_rec+len(sgr_temp)
                #N_gr_rec=N_gr_rec+len(rec_phase)
                #Plot Dst
                if pltt is not None:
                    am.simple_plot(rec_phase,sgr[rec_phase],"Hourly index","Sgr","Recovery Phase Sgr",show=1)
        #print("Iteration"+str(ii))
    
    
    #Flatten nested Dst end eflux lists to simple lists
    sgeo_main=am.flatten_list(sgeo_main)
    #For checking that flatten_array works properly
    #sgeo_main_1=am.flatten_list(sgeo_main)
    # kk=0
    # for ii in range(len(sgeo_main)):
    #     temp1=list(sgeo_main[ii])
    #     temp2=sgeo_main_1[kk:kk+len(temp1)]
    #     kk=kk+len(temp1)
    #     if temp1!=temp2:
    #         print("temp1!=temp2")
    sgr_main=am.flatten_list(sgr_main)
    sgeo_rec=am.flatten_list(sgeo_rec)
    sgr_rec=am.flatten_list(sgr_rec)
    eflux_sgeo_main=am.flatten_list(eflux_sgeo_main)
    eflux_sgeo_rec=am.flatten_list(eflux_sgeo_rec)
    eflux_sgr_main=am.flatten_list(eflux_sgr_main)
    eflux_sgr_rec=am.flatten_list(eflux_sgr_rec)
    
    
    #Evaluate mutul information
    mi_sgeo_main=obj_sgeo._mutual_info(sgeo_main,eflux_sgeo_main)
    mi_sgeo_rec=obj_sgeo._mutual_info(sgeo_rec,eflux_sgeo_rec)
    mi_sgr_main=obj_sgr._mutual_info(sgr_main,eflux_sgr_main)
    mi_sgr_rec=obj_sgr._mutual_info(sgr_rec,eflux_sgr_rec)
    mi_err_sgeo_main=obj_sgeo._mi_shuffle(sgeo_main,eflux_sgeo_main)
    mi_err_sgr_main=obj_sgeo._mi_shuffle(sgr_main,eflux_sgr_main)
    mi_err_sgeo_rec=obj_sgr._mi_shuffle(sgeo_rec,eflux_sgeo_rec)
    mi_err_sgr_rec=obj_sgr._mi_shuffle(sgr_rec,eflux_sgr_rec)
    
    #Plot data points from the events
    if draw is not None:
        xlabel="log10 of e flux"
        y_label_geo="log10 of Sgeo spectral power"
        y_label_gr="log10 of Sgr spectral power"
        ttl_geo_rec=str("Storm rec phases for Sgeo, e flux type"+e_flux_type)
        ttl_gr_rec=str("Storm rec phases for Sgr, e flux type"+e_flux_type)
        ttl_geo_main=str("Storm main phases for Sgeo, e flux type"+e_flux_type)
        ttl_gr_main=str("Storm main phases for Sgr, e flux type"+e_flux_type)
        am.simple_scatter(sgeo_main,eflux_sgeo_main,xlabel,y_label_geo,ttl_geo_main,show=1,lin_fit=1)
        am.simple_scatter(sgeo_rec,eflux_sgeo_rec,xlabel,y_label_geo, ttl_geo_rec,show=1,lin_fit=1)
        am.simple_scatter(sgr_main,eflux_sgr_main,xlabel,y_label_gr,ttl_gr_main,show=1,lin_fit=1)
        am.simple_scatter(sgr_rec,eflux_sgr_rec,xlabel,y_label_gr,ttl_gr_rec,show=1,lin_fit=1)
    
    #Correlations
    sgeo_main_corr=stats.pearsonr(sgeo_main,eflux_sgeo_main)
    sgeo_rec_corr=stats.pearsonr(sgeo_rec,eflux_sgeo_rec)
    sgr_main_corr=stats.pearsonr(sgr_main,eflux_sgr_main)
    sgr_rec_corr=stats.pearsonr(sgr_rec,eflux_sgr_rec)
    
    #Plot Dst, ULF and e fluxes from the Rostoker events
    # plt.plot(dates[24768:25392],sgeo[24768:25392]);plt.title("sgeo 1993");plt.show();
    # plt.plot(dates[24768:25392],eflux[24768:25392]);plt.title("eflux 1993");plt.show();
    # plt.plot(dates[24768:25392],dst[24768:25392]);plt.title("dst 1993");plt.show()
    
    # plt.plot(dates[27744:29928],sgeo[27744:29928]);plt.title("sgeo 1994");plt.show();
    # plt.plot(dates[27744:29928],eflux[27744:29928]);plt.title("eflux 1994");plt.show();
    # plt.plot(dates[27744:29928],dst[27744:29928]);plt.title("dst 1994");plt.show()
    
    #Print main results
    if prnt_result is not None:
        print("e flux type is %s" %e_flux_type)
        print("%i data points evaluated for Sgeo main phase" %n_sgeo_main)
        print("%i data points evaluated for Sgr main phase" %n_sgr_main)
        print("%i data points evaluated for Sgeo recovery phase" %n_sgeo_rec)
        print("%i data points evaluated for Sgr recovery phase" %n_sgr_rec)
        print("I(Sgeo main, e) = "+str(round(mi_sgeo_main,5))+" noise="+str(round(mi_err_sgeo_main[0],5))+" sigma = "+str(round(mi_err_sgeo_main[1],5)))
        print("I(Sgr main, e) = "+str(round(mi_sgr_main,5))+" noise="+str(round(mi_err_sgr_main[0],5))+" sigma = "+str(round(mi_err_sgr_main[1],5)))
        print("I(Sgeo rec, e) = "+str(round(mi_sgeo_rec,5))+" noise="+str(round(mi_err_sgeo_rec[0],5))+" sigma = "+str(round(mi_err_sgeo_rec[1],5)))
        print("I(Sgr rec, e) = "+str(round(mi_sgr_rec,5))+" noise="+str(round(mi_err_sgr_rec[0],5))+" sigma = "+str(round(mi_err_sgr_rec[1],5)))
        
        print("corr(Sgeo main, e = "+str(round(sgeo_main_corr[0],5))+"p = "+str(round(sgeo_main_corr[1],5)))
        print("corr(Sgr main, e = "+str(round(sgr_main_corr[0],5))+"p = "+str(round(sgr_main_corr[1],5)))
        print("corr(Sgeo rec, e = "+str(round(sgeo_rec_corr[0],5))+"p = "+str(round(sgeo_rec_corr[1],5)))
        print("corr(Sgr rec, e = "+str(round(sgr_rec_corr[0],5))+"p = "+str(round(sgr_rec_corr[1],5)))
    
    #Return mutual information and errors: one row contains mi, error and sigma
    #First row is Sgeo main phase, the follpowing ones are
    #Sgr main phase
    #Sgeo recovery  phase
    #Sgr recovery phase
    if ret_result is not None:    
        result=np.array([[mi_sgeo_main,mi_err_sgeo_main[0],mi_err_sgeo_main[1]],
                         [mi_sgr_main,mi_err_sgr_main[0],mi_err_sgr_main[1]],
                         [mi_sgeo_rec,mi_err_sgeo_rec[0],mi_err_sgeo_rec[1]],
                         [mi_sgr_rec,mi_err_sgr_rec[0],mi_err_sgr_rec[1]]])
        return result


#Set storm parameters
storm_param=[-50,-50,-5,10]
dst_onset=storm_param[0]
dst_rec=storm_param[1]
bz_limit=storm_param[2]
dst_pre_tt=storm_param[3]
e_flux_type='Fe130'
#Find the storm events from the data
path="/Users/mikkosavola/Documents/Opiskelu_HY/Gradu/"\
                "raw_data/ViRBO_Space_science/Simms_replication"
path.replace(" ", "")
path.replace("\n", "")
os.chdir(path)
file1=am.create_path(path,'ViRBO_ULF.txt')
file2=am.create_path(path,'e_flux.txt')

obj_dst,obj_sgr,obj_sgeo,storms=fst.find_storms(file1,file2,dst_onset,dst_rec,bz_limit,dst_pre_tt,e_flux_type)
#Set time offest to 0 for this case
t_offset=0
#Put storm data, into a list
storm_data=[obj_dst,obj_sgr,obj_sgeo,storms]

simms_analysis_2(storm_param,storm_data,e_flux_type,t_offset,draw=1,prnt_result=1)