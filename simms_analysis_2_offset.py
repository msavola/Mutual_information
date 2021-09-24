#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:47:52 2020

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
from simms_analysis2_new import simms_analysis_2 as sa2

def simms_analysis_2_offset(storm_data,storm_param,e_flux_type,t_offset):
    #Runs the Simms analysis 2 for desired time offset
    e_flux_type=e_flux_type
    
    I=np.empty([4,3,t_offset+1])
    times=np.linspace(0,t_offset,t_offset+1,dtype=np.int64)
    for jj in range(len(times)):
        print("\nIteration %i out of %i"%(jj,t_offset+1))
        I[:,:,jj]=sa2(storm_param,storm_data,e_flux_type,times[jj],prnt_result=1,ret_result=1,draw=None)
        
        
    #Plot results
    x_label="time offset, ULF precedes e flux (h)"
    y_label_geo="Mutual information (bits)"
    y_label_gr="Mutual information (bits)"
    ttl_geo_rec=str("I(Sgeo, e flux "+e_flux_type+"), recovery phase")
    ttl_gr_rec=str("I(Sgr, e flux "+e_flux_type+"), recovery phase")
    ttl_geo_main=str("I(Sgeo, e flux "+e_flux_type+"), main phase")
    ttl_gr_main=str("I(Sgr, e flux "+e_flux_type+"), main phase")
    
    #Sgeo main
    #am.simple_scatter(times,I[0,0,:],x_label,y_label_geo,ttl_geo_main,legend=None,show=None)
    am.plot_w_noise(times,I[0,0,:],I[0,1,:],I[0,2,:],3,x_label,y_label_geo,"measurement data","noise",ttl_geo_main)
    plt.show()
    #Sgr main
    #am.simple_scatter(times,I[1,0,:],x_label,y_label_geo,ttl_gr_main,legend=None,show=None)
    am.plot_w_noise(times,I[1,0,:],I[1,1,:],I[1,2,:],3,x_label,y_label_gr,"measurement data","noise",ttl_gr_main)
    plt.show()
    #Sgeo recovery
    #am.simple_scatter(times,I[2,0,:],x_label,y_label_geo,ttl_geo_rec,legend=None,show=None)
    am.plot_w_noise(times,I[2,0,:],I[2,1,:],I[2,2,:],3,x_label,y_label_geo,"measurement data","noise",ttl_geo_rec)
    plt.show()
    #Sge recovery
    #am.simple_scatter(times,I[3,0,:],x_label,y_label_gr,ttl_gr_rec,legend=None,show=None)
    am.plot_w_noise(times,I[3,0,:],I[3,1,:],I[3,2,:],3,x_label,y_label_gr,"measurement data","noise",ttl_gr_rec)
    plt.show()
    
   
    
#Set storm parameters
storm_param=[-30,-30,-5,10]
dst_onset=storm_param[0]
dst_rec=storm_param[1]
bz_limit=storm_param[2]
dst_pre_tt=storm_param[3]
e_flux_type='Fe1p2'
t_max=4*24 #max offset, in hours for this implementation

#Find the storm events from the data
path="/Users/mikkosavola/Documents/Opiskelu_HY/Gradu/"\
                "raw_data/ViRBO_Space_science/Simms_replication"
path.replace(" ", "")
path.replace("\n", "")
os.chdir(path)
file1=am.create_path(path,'ViRBO_ULF.txt')
file2=am.create_path(path,'e_flux.txt')

obj_dst,obj_sgr,obj_sgeo,storms=fst.find_storms(file1,file2,dst_onset,dst_rec,bz_limit,dst_pre_tt,e_flux_type)

#Put storm data, into a list
storm_data=[obj_dst,obj_sgr,obj_sgeo,storms]    

simms_analysis_2_offset(storm_data,storm_param,e_flux_type,t_max)