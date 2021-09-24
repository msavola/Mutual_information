#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:12:10 2020

@author: mikkosavola
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.stats as stats
from itertools import dropwhile
from itertools import groupby
import os
import sys
import numpy.polynomial.polynomial as poly

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

def figstodir(fesafile):
    directory = "plots/FESA"+finder(fesafile).replace(".","_")   
    parent_dir = "/Users/mikkosavola/Documents/Opiskelu_HY/Kandidaatintutkielma" #Parent Directory path 
    path = os.path.join(parent_dir, directory)  # Path 
  
    try:
        os.makedirs(path) #Create directory
        print("Directory '% s' created" % directory) 
        return path
    except OSError as error:
        print("Folder",path,"exists already")
        return path
    
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
    
def moving_average(x,step):
    #Evaluates the moving average of the vector x in sets of size step
    narr=[]
    for ii in range(len(x)-step+1):
        x_temp=x[ii:ii+step]
        narr.append(np.nanmean(x_temp)) 
    return np.asarray(narr,dtype=np.float64)
    





##################
    #FUNCTIONS FOR PLOTTING

def plot_w_noise(x,y,noise,sigma,nsigma,xlabel,ylabel,lbl_1,lbl_2,ttl):
    #Plot
    plt.plot(x,y,'bo',label=lbl_1)
    txt=str("Noise with "+str(nsigma)+" confidence interval")
    plt.plot(x,noise,color="orange",label=lbl_2)
    plt.fill_between(x,noise-nsigma*sigma,noise+nsigma*sigma,label=txt, color='orange', alpha=.1)
    add_labels(xlabel,ylabel,ttl)
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
    
def simple_plot(x,y,x_label,y_label,ttl,show=None,save=None,lin_fit=None):
    if lin_fit is not None:
        linear_fit(x,y,1,plot=1)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(ttl)
    plt.legend()
    if save is not None:
        ttl.replace(" ","_")
        plt.savefig(str(ttl+".pdf"),format="pdf")
    if show is not None:
        plt.show()
        
def simple_scatter(x,y,x_label,y_label,ttl,legend=None,show=None,save=None,lin_fit=None):
    if lin_fit is not None:
        linear_fit(x,y,1,plot=1)
    plt.scatter(x,y,label="measurement data")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(ttl)
    if legend is not None:
        plt.legend()
    if save is not None:
        ttl.replace(" ","_")
        plt.savefig(str(ttl+".pdf"),format="pdf")
    if show is not None:
        plt.show()

def linear_fit(x,y,order,plot=None):
    #Fit a polynomail of order to x and y values
    coefs=poly.polyfit(x,y,order)
    x_new=np.linspace(min(x),max(x)+1,num=len(x)*10)
    ffit=poly.polyval(x_new,coefs)
    #Calculate correlation of data sets
    corrs=stats.pearsonr(x,y)
    corr=round(corrs[0],3)
    p_val=round(corrs[1],3)
    txt=str("correlation = "+str(corr)+"\ntwo-tailed p value of "+str(p_val))
    if plot is not None:
        plt.plot(x_new,ffit,color="orange",label="least-square linear fit")
        plt.text(0.5, -0.5, txt, horizontalalignment='center',verticalalignment='center',bbox=dict(alpha=0.5))
    return
    
    
######################
#BELOW ARE ADDITIONAL FUNCTIONS FOR THE SIMMS STUDY
def find_min_dst(x,n,k,limit):
    #Finds the minimum Dst in a -n/2 to n/2-1 hour window
    #If the minimum is among the last k<=n-1 values, the search is continued,
    #because a smaller minimum might be coming up in the next n values;
    #if not, the found minimum will be caught in the next search
    if k>0.5*n-1:
        print("Your k parameter is larger than n-1, please give a smaller k")
        return False
    mini=min(x)
    #Discard if mini among k last values
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
    if all_nans(x):
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
    #Continue until the monotonous series ends, the found Dst is above the threshold
    #and the preceding Dst values are strictly smaller than the maximum
    while (x[ind]!=limit or dst[ind+1]<dst_limit or not np.all(dst[ind-10:ind+1]<dst[ind+1])):
    #while (x[ind]!=limit or dst[ind+1]<dst_limit):
        ind=ind-1
    ind=ind+1
    return ind
        
    
def find_after_storm():
    #Store the post storm phase related to a Dst drop
    1
    
def find_recovery(dst,dst_limit):
    #Check that Dst has recovered above the dst_limit
    #Returns the relative index number of dst recovery
    ind_rec=np.where(dst>dst_limit)[0][0]
    return ind_rec
    
# def check_recovery(x,end,threshold):
#     #Check that there is at least 72 hours between storms
#     if sum(x[end:end+threshold])>1:
#         return False
    
def end_of_recovery(x,y,time):
    #Find whether the after storm recovery phase's end exceeds the data sets
    #length and choose the appropriate value
    for ii in range(len(x)):
        if x[ii]+time>len(y):
            x[ii]=len(y)
        else:
            x[ii]=x[ii]+time
    return x

def zip_list(x,y):
    #Return a zipped list
    result=zip(x,y)
    result=list(result)

    return result

def contains_nan(x):
    #Return True is x contains at least one nan
    result=False
    if not is_np_array(x):
        x=[x]
        x=np.asarray(x,dtype=np.float64)
    for ii in range(len(x)):
        if x[ii]=="nan" or x[ii] is np.nan or x[ii] is float('nan') or np.isnan(x[ii]):
        #(x[ii]=="nan" or x[ii] is np.nan or x[ii] is float('nan') or np.isnan(x[ii])):
            result=True
            return result
    return result

def all_nans(x):
    #Return True if x contains only nans
    #Convert x to list to handle scalars
    if not is_np_array(x):
        x=[x]
        x=np.asarray(x,dtype=np.float64)
    for ii in range(len(x)):
        if x[ii]=="nan" or x[ii] is np.nan or x[ii] is float('nan') or np.isnan(x[ii]):
        #(x[ii]=="nan" or x[ii] is np.nan or x[ii] is float('nan') or np.isnan(x[ii])):
            result=True
        else:
            result=False
            return result            
    return result

def recurring_values(x,n):
    #Check whether some value in x recurs at least n times in succession
    grouped_x=[(k,sum(1 for ii in g)) for k,g in groupby(x)]
    grouped_x=np.asarray(grouped_x)
    if np.all(grouped_x[:,1]<n):
        return False
    else:
        return True
    
def is_list(x):
    if type(x)==list:
        return True
    
def is_np_array(x):
    if type(x)==np.array or type(x)==np.ndarray:
        return True
    

def flatten_list(l, ltypes=(list, tuple)):
    #Flattens a nested list of list or numpy arrays or numpy
    #ndarrays into a one-dimensional list
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        if is_np_array(l[i]):
            l[i]=list(l[i])
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def create_path(path,file):
    #Concatenates the directory and the file name
    #into a single path
    if type(path)!=str:
        print("Directory name must be a string")
        exit
    if type(file)!=str:
        print("File name must be a string")
        sys.exit()
    if path[-1]!='/':
        path=str(path+'/')
    new_path=str(path+'/'+file)
    return new_path

def _zero_array(x):
    #Create a 2D array of zeros with columns equal to the nbumber of input
    #and rows equal to the length of one input array
    cols=len(x)
    rows=len(x[0])
    return np.zeros([rows,cols])

def _elements_equal(*args):
    #Takes a number of equal length arrays and checks,
    #how many of the indices are equal
    #Collect inputs into columns in an array
    arrs=_zero_array(args)
    ii=0
    for arg in args:
        arrs[:,ii]=arg
        ii=ii+1
    
    rows=np.shape(arrs)[0]
    #Check how many times array elements are equal
    summa=0
    for ii in range(rows):
        col_0=arrs[ii,0]
        if np.all(arrs[ii,:]==col_0):
            summa=summa+1
    
    return summa
        

# def flatten_nparray(l, ltypes=(np.array, tuple)):
#     ltype = list
#     l = list(l)
#     i = 0
#     while i < len(l):
#         while isinstance(l[i], ltypes):
#             if not np.all(l[i]):
#                 l.pop(i)
#                 i -= 1
#                 break
#             else:
#                 l[i:i + 1] = l[i]
#         i += 1
#     return ltype(l)

# def flatten_npndarray(l, ltypes=(np.ndarray, tuple)):
#     ltype = list
#     l = list(l)
#     i = 0
#     while i < len(l):
#         while isinstance(l[i], ltypes):
#             if not np.all(l[i]):
#                 l.pop(i)
#                 i -= 1
#                 break
#             else:
#                 l[i:i + 1] = l[i]
#         i += 1
#     return ltype(l)

   