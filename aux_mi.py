#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:12:10 2020

@author: mikkosavola
"""

import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import dropwhile
import os

#Functions for reading in data, writing into file and saving figures
def new_comment(f,txt):
#Write txt as a comment line in an open file f
    _txt="#"+txt+"\n"
    f.write(_txt)
    
def is_comment(s):
    #function to check if a line starts with some character, here "#" for comment
    # return true if a line starts with #
    return s.startswith('#')

def reader_ULF(name):
#This function reads ULF data from a .txt file into an array
    file = open(name, 'r')
    fname=name
    arr1=[]
    arr2=[]
    np.set_printoptions(precision=16) #For checking manually that input data is not rounded
    
    for line in dropwhile(is_comment,file):
        pl=line.split()
        arr1.append(np.unicode_(pl[0]))
        arr2.append(np.float64(pl[1]))               
    #Create a record of the arrays
    rec=np.rec.fromarrays((arr1, arr2), names=('dates', 'power'))
    file.close()
    return rec,fname

def reader_FESA(name):
#This function reads FESA data from a .txt file into an array
    file = open(name, 'r')
    fname=name
    arr1=[]
    arr2=[]
    np.set_printoptions(precision=16) #For checking manually that input data is not rounded
    
    for line in dropwhile(is_comment,file):
        pl=line.split()
        arr1.append(np.float32(pl[0])) #L bin
        arr2.append(np.float64(pl[1:]))#The row of flux values per time stampt for the respective L bin              
    file.close()
    arr1=np.asarray(arr1)
    arr2=np.asarray(arr2)
    return arr1, arr2, fname

def reader_VIRBO(name):
#This function reads ViRBO data from a .txt file into an array
    file = open(name, 'r')
    fname=name
    arr1=[]
    arr2=[]
    arr3=[]
    arr4=[]
    arr5=[]
    arr6=[]
    arr7=[]
    arr8=[]
    arr9=[]
    arr10=[]
    arr11=[]
    arr12=[]
    arr13=[]
    arr14=[]
    arr15=[]
    np.set_printoptions(precision=16) #For checking manually that input data is not rounded
    
    for line in dropwhile(is_comment,file):
        pl=line.split()
        arr1.append(np.unicode_(pl[0]))
        arr2.append(np.int64(pl[1]))
        arr3.append(np.int64(pl[2]))
        arr4.append(np.int64(pl[3]))
        arr5.append(np.int64(pl[4]))
        arr6.append(np.int64(pl[5]))
        arr7.append(np.float64(pl[6]))
        arr8.append(np.float64(pl[7]))
        arr9.append(np.float64(pl[8]))
        arr10.append(np.float64(pl[9]))
        arr11.append(np.float64(pl[10]))
        arr12.append(np.float64(pl[11]))
        arr13.append(np.float64(pl[12]))
        arr14.append(np.float64(pl[13]))
        arr15.append(np.float64(pl[14]))
    #Create a record of the arrays
    rec=np.rec.fromarrays((arr1, arr2,arr3,arr4,arr5,arr6,arr7,arr8,
                           arr9,arr10,arr11,arr12,arr13,arr14,arr15),
                          names=('year', 'month','day','hour','min','sec',
                                 'Tgeo','Sgeo','Rgeo','Tgr','Sgr',
                                 'Rgr','Dst','V','Bz'))
    file.close()
    return rec,fname

def reader_EFLUX(name):
#This function reads e flux data from a .txt file into an array. The data (Fe130 and Fe1p2)
#is described in of Borosky&Yakymenko 2017 in sections 2.2 and 2.3
    file = open(name, 'r')
    fname=name
    arr1=[]
    arr2=[]
    arr3=[]
    arr4=[]
    arr5=[]
    np.set_printoptions(precision=16) #For checking manually that input data is not rounded
    
    for line in dropwhile(is_comment,file):
        pl=line.split()
        arr1.append(np.int64(pl[0])) 
        arr2.append(np.int64(pl[1]))  
        arr3.append(np.int64(pl[2])) 
        arr4.append(np.float64(pl[3])) 
        arr5.append(np.float64(pl[4])) 
    #Create a record of the arrays
    rec=np.rec.fromarrays((arr1,arr2,arr3,arr4,arr5),
                          names=('year', 'doy','hour','Fe1p2','Fe130'))
    file.close()
    return rec, fname

def finder(name1):
#Finds the FESA energy range provided in the file name
    word1="MeV"
    lword1=len(word1)
    ind=name1.find(word1)
    res=name1[ind-3:ind+lword1]
    import pdb; pdb.set_trace()
    return res
    
def transf_name(_ufunc):
    #Returns the "name" of a utility function ufunc when given str(ufunc) as input
        n_ufunc=str(_ufunc)
        begin=n_ufunc.find("<", 0, len(n_ufunc))
        end=n_ufunc.find(">", 0, len(n_ufunc))
        _name=n_ufunc[begin+8:end-1]
        return _name
    
def write_to_file(f,x):
#Writes vector x as lines into a file
    for ii in range(len(x)):
         f.write(str(x[ii])+"\n")
    print("Saved data into file")

def annot(xs,ys,zs,text):
    # zip joins x, y and z coordinates in triplets
    #xs and ys give the points to be annotated, zs contains the annortations and text is
    #the text to be added
    for x,y,z in zip(xs,ys,zs):
        label = str(text+" "+"{:.2f}".format(z))
        plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
        
def shuffler(x):
#Shuffle the values of vectos x in random ordering
    lx=len(x)
    nx=random.sample(list(x),lx)
    return np.asarray(nx)

def hours_avg(arr1,hours):
#Calculates the time average of arr1, in chunks of hours. E.g. if arr1 is 1x20 and hours equals 5
#The averages calculated are arr1[0:5], arr1[5:10], arr1[10:15] and arr1[15:20].
#It is assumed that arr1's length is a multiple of step.
    N=int(len(arr1)/hours)
    if len(arr1)%hours != 0:
        print('The given array is not a multiple of the given step size.')
    else:
        narr=[]
        for ii in range(N):
            start=ii*hours
            end=(ii+1)*hours
            narr.append(np.nanmean(arr1[start:end]))
        return np.asarray(narr,dtype=np.float64)
    





##################
    #FUNCTIONS FOR PLOTTING

def plot_w_noise(x,y,sigma,nsigma,lbl,ttl):
    #Plot
    plt.plot(x,y,'bo',label=lbl)
    txt=str("Noise with "+str(nsigma)+" confidence interval")
    plt.plot(x,y,color="orange",label=lbl)
    plt.fill_between(x,y-nsigma*sigma,y+nsigma*sigma,label=txt, color='orange', alpha=.1)
    add_labels("time offset (steps)", "mutual information (bits)",ttl)
    plt.legend

def add_labels(xlbl,ylbl,ttl):
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.title(ttl)

def two_y_plot(x,y1,y2,xlbl,y1_lbl,y2_lbl,ttl,txt=None,save=None):
    #Plot data with two y axes
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel(xlbl)
    ax1.set_ylabel(y1_lbl, color=color)
    ax1.plot(x,y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel(y2_lbl, color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #Currently manually change info in the title and txt
    plt.title(ttl)
    if txt is not None:
        txt=txt
    ax1.text(0.1, 0.15, txt, transform=ax1.transAxes, fontsize=8,
        verticalalignment='top')
    if save is not None:
        plt.savefig(str(ttl+".pdf"),format="pdf")
    plt.show()
    
    
######################
#BELOW ARE ADDITIONAL FUNCTIONS FOR THE SIMMS STUDY
def find_min_dst(x,n,k,limit):
    #Finds the minimum Dst in a -n to +n-1 hour window
    #If the minimum is among the last k<=n-1 values, the search is continued,
    #because a smaller minimum might be coming up in the next n values;
    #if not, the found minimum will be caught in the next search
    if k>0.5*n-1:
        print("Your k parameter is larger than n-1, please give a smaller k")
        return False
    mini=min(x)
    #Discard if mini among five last values
    discard=sum(mini==x[-k:])
    if discard==0:
        if mini in x and dst_below(mini,limit):
            mini_index = np.where(x == mini)[0] #get index of mini 
        else:
            return False
    else:
        return False
    #If multiple values, take the last one
    if len(mini_index)>1:
        mini_index=max(mini_index)
    #print(mini_index)
    return int(mini_index)

def dst_below(x,limit):
    #Checks whether the Dst value is below the limit
    if x<limit:
        return True
    
def check_min_bz(x,limit):
    #Check if x consists of nans only
    if np.all(x=='nan' or x is np.nan or x is float('nan') or np.isnan(x)):
        return False
    
    if np.nanmin(x)<limit:
        return True
    else:
        return False
    
def find_pre_storm(dur_pre,end,x):
    #Store the pre storm phase related to a Dst drop
    1

def decreasing_sequences(x):
    #Creates a vector that has 1 at each entry, where the corresponding entry
    #in x is larger than the following one
    seqs=np.empty(len(x))
    for ii in range(len(seqs)-1):
        if x[ii+1]<x[ii]:
            seqs[ii]=1
        else:
            seqs[ii]=0
    return seqs
    
def find_main_phase(x,start,limit,dst,dst_limit):
    #Find the beginning of the main phase based on the occurrence of the
    #Dst drop
    #Goal is the value at which the search will end, and start is the starting
    #index for the search (use absolute index of x)
    #ind is iterated
    ind=start-1
    while (x[ind]!=limit or dst[ind]<dst_limit):
        ind=ind-1
    ind=ind+1
    return ind
        
    
def find_after_storm():
    #Store the post storm phase related to a Dst drop
    1
    
def find_recovery(dst,dst_limit,t):
    #Check that Dst has sustainably recovered above the dst_limit
    #Dst must be above dst_limit t time steps after the first occurrence
    #of dst exceeding the dst_limit
    #Returns the relative index number of dst recovery
    ii=0
    _found=False
    while _found is False: 
        ind_rec=np.where(dst[ii*t:]>dst_limit)[0][0]
        ii=ii+1
        if np.all(dst[ind_rec+(ii-1)*t:ind_rec+ii*t]>dst_limit):
            _found=True
    ind_rec=ind_rec+ii*t
    return ind_rec
    
def check_recovery(x,end,threshold):
    #Check that there is at least 72 hours between storms
    if sum(x[end:end+threshold])>1:
        return False
    
def end_of_recovery(x,y,time):
    #Find whether the after storm recovery phase's end exceeds the data sets
    #length and choose the appropriate value
    for ii in range(len(x)):
        if x[ii]+time>len(y):
            x[ii]=len(y)
        else:
            x[ii]=x[ii]+time
    return x
        
