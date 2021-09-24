#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:12:06 2020

@author: mikkosavola
"""

import numpy as np
#import scipy
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) #adjusts figure sizes automatically
plt.rcParams['figure.dpi'] = 300 #set plot dpi to 300
#import numpy.random as ran
import scipy.stats as stats
#import datetime as dt
import aux_mi as am

#Class for creating objects to handle data and calculate pdfs and mutual information
class My_Mi:
    def __init__(self,file1,dates,x,file2,Lshells,y):
        #ULF data
        self.dates=dates
        self.x=x
        #FESA data
        self.Lshells=Lshells
        self.y=y
        #filenames
        self.file1=file1
        self.file2=file2
        #maxima and minima
        # self.maxx=max(x)
        # self.minx=min(x)
        # self.maxy=max(y)
        # self.miny=min(y)
        # #lengths
        # self.lenx=len(x)
        # self.leny=len(y)
        
    def _n_bins(self,x,fixed_bins=None):
    #Function to define the number of bins for functions _create_pdf and _joint_pdf
    #fixed_bins can be given to fix the number of bins
        iqr=stats.iqr(x)
        lx=len(x)
        #Check if the vector has only constant values or if iqr==0
        if (np.all(np.floor(x)==np.ceil(x)) and lx==1) or iqr==0: 
            bins=1
        elif fixed_bins is not None:
            bins=fixed_bins
        else:
            #bins=int(np.ceil(lx**(1/3))) #cube-root of number of data points
            #remove comments below to use Freedman-Doaconis instead
            xma=np.max(x)
            xmi=np.min(x)
            bins=((xma-xmi)*lx**(1/3))/(2*iqr)    #Freedman-Diaconis rule
            bins=int(np.ceil(bins))
            #import pdb; pdb.set_trace()
        
        return bins
    
    def _bin_index(self,x,mini,maxi,steps):
    #Return bin index of x
        step=(maxi-mini)/steps
        rem1=x-mini #Get remainder
        if rem1==0 and step==0:
            S1=1
        else:
            S1=np.floor(rem1/step) #Number of full steps from mini is the index of the respective bin
        S1=int(S1)
        if S1==steps: #Adress the maximum value in data and insert it into the largest bin
            S1=S1-1
        return S1 #Determine the bin to which the ii entry in x belongs to 
    
    
    def _bin_occurrences(self,x):
    #Assign number of occurrences to each bin
        mini=np.min(x)
        maxi=np.max(x)
        steps=self._n_bins(x)
        N=np.zeros(steps)
        for ii in range(len(x)):
            S=self._bin_index(x[ii],mini,maxi,steps)
            N[S]=N[S]+1 #Increase the number of occurrences in the bin by 1
        return N #Return the vector containing the occurrences per bin
    
    def _create_pdf(self,x,draw=None,dname=None,xaxis=None,yaxis=None):
    #The function creates a probability density function of given data
    #based on the partition, which is the number of brackets
    #It is assumed that the data is a function of an underlying variable, e.g. time, 
    #and the data points are in increasing order of the underlying variable
    #Set minimum, maximum, stepsize and array for calculating number of occurrences of each bracket 
        #print("Calculating pdf")
        #import pdb; pdb.set_trace()
        lx=len(x)
        mini=np.min(x)
        maxi=np.max(x)
        partition=self._n_bins(x)
        step=(maxi-mini)/partition
        if step==0:
            step=1 #Address vectors with constant values only
    
        N=self._bin_occurrences(x)         
        #Normalize probabilities
        pdf=N/lx
            
        
        brackets=np.linspace(mini+step/2,maxi-step/2,num=partition) #For plotting only, set mid-point of each bin
        #REMOVE COMMENTS IF HISTOGRAM IS DESIRED
        #bracketslim=np.linspace(mini,maxi,num=partition+1) #"Walls" of the bins
        #fig=plt.figure()
        #ax=fig.gca()
        #ax.set_xticks(np.arange(mini,maxi,partition+1)) #Set vertical lines to illustrate brackets
        #ax.plot(brackets,N,'ro')
        #plt.title("Number of events per bracket")
        #plt.grid(True)
        #plt.show()
        
        if draw is not None:
            #Plot distribution
            plt.plot(brackets,pdf,'ro')
            text=str("Code created pdf from "+str(dname)+" data with "+str(partition)+" bins for "+str(len(x))+" data points")
            plt.title(text)
            yadd=1.1
            plt.ylim(min(pdf)*yadd,max(pdf)*yadd)
            plt.xlabel(xaxis)
            plt.ylabel(yaxis)
            plt.savefig(text+".pdf",format="pdf")
            plt.show()
    
        #print("The marginal pdf is")
        #print(pdf)
        return pdf

    
    def _joint_pdf(self,x,y):
    #The function creates a joint probability density function of two given data sets.
    #Set minimum, maximum, stepsize and array for calculating number of occurrences of each bin 
        #print("Calculating joint pdf")
        mini1=np.min(x)
        maxi1=np.max(x)
        mini2=np.min(y)
        maxi2=np.max(y)
        part1=self._n_bins(x) #partitions
        part2=self._n_bins(y)
        step1=(maxi1-mini1)/part1
        step2=(maxi2-mini2)/part2
        lx=len(x)
        ly=len(y)
        if step1==0:
            step1=1 #Address vectors with constant values only
        if step2==0:
            step2=1 #Address vectors with constant values only
        bracs1=np.zeros(lx) #Bin indices for data1
        bracs2=np.zeros(ly) #Bin indices for data2
        pdf=np.zeros([part1,part2]) #Matrix for joint probabilities
    
        #Assign bin indices
        for ii in range(lx):
            bracs1[ii]=self._bin_index(x[ii],mini1,maxi1,part1)
        for jj in range(ly):    
            bracs2[jj]=self._bin_index(y[jj],mini2,maxi2,part2)
    
        #Populate pdf
        Total=0
        for ii in range(0,part1):
            test1=1*(bracs1==ii)
            for jj in range(0,part2):
                test2=1*(bracs2==jj)
                summa=np.sum(test1.dot(test2))
                Total=Total+summa
                #import pdb; pdb.set_trace()
                pdf[ii][jj]=summa
        pdf=pdf/Total #Normalize
        #print("The sum of the joint probabilities is equal to", np.sum(pdf))
        #print(pdf)
        return pdf
    
    def _joint3_pdf(self,x,y,z):
    #The function creates a joint probability density function of three given data sets.
    #Set minimum, maximum, stepsize and array for calculating number of occurrences of each bin 
        #print("Calculating joint pdf")
        mini1=np.min(x)
        maxi1=np.max(x)
        mini2=np.min(y)
        maxi2=np.max(y)
        mini3=np.min(z)
        maxi3=np.max(z)
        #partitions
        part1=self._n_bins(x) 
        part2=self._n_bins(y)
        part3=self._n_bins(z) #Change this back to default, after fixed_bins is used
        step1=(maxi1-mini1)/part1
        step2=(maxi2-mini2)/part2
        step3=(maxi3-mini3)/part3
        lx=len(x)
        ly=len(y)
        lz=len(z)
        #Address input vectors with constant values only
        if step1==0:
            step1=1 
        if step2==0:
            step2=1
        if step3==0:
            step3=1
        #Assign bin indices
        bracs1=np.zeros(lx)
        bracs2=np.zeros(ly)
        bracs3=np.zeros(lz)
        #Matrix for joint probabilities
        pdf=np.zeros([part1,part2,part3])
    
        #Assign bin indices
        for ii in range(lx):
            bracs1[ii]=self._bin_index(x[ii],mini1,maxi1,part1)
        for jj in range(ly):    
            bracs2[jj]=self._bin_index(y[jj],mini2,maxi2,part2)
        for kk in range(lz):    
            bracs3[kk]=self._bin_index(z[kk],mini3,maxi3,part3)
    
        #Populate pdf
        Total=0
        for ii in range(0,part1):
            test1=1*(bracs1==ii)
            for jj in range(0,part2):
                test2=1*(bracs2==jj)
                for kk in range(0,part3):
                    test3=1*(bracs3==kk)
                    summa=am._elements_equal(test1,test2,test3)
                    Total=Total+summa
                    #import pdb; pdb.set_trace()
                    pdf[ii][jj][kk]=summa
        pdf=pdf/Total #Normalize
        #print("The sum of the joint probabilities is equal to", np.sum(pdf))
        #print(pdf)
        return pdf    
    
    def _mutual_info(self,x,y):
    #Calculate the mutual information based on two given data sets
        x,y=self._remnans(x,y)
        y,x=self._remnans(y,x)
        pdfX=self._create_pdf(x)
        pdfY=self._create_pdf(y)
        pdfXY=self._joint_pdf(x,y)
        I=0
        for ii in range(len(pdfX)):
            for jj in range(len(pdfY)):
                #Handle division by zero and other exceptions
                if (pdfX[ii]*pdfY[jj])==0 and pdfXY[ii][jj]>0:
                    I=100 #This is a convention, see "Elements of Infomation Theory" OR SET TO SOME LARGE FINITE NUMBER
                    return I
                elif (pdfX[ii]*pdfY[jj])==0 and pdfXY[ii][jj]==0:
                    In=0 #ibid.
                elif (pdfX[ii]*pdfY[jj])!=0 and pdfXY[ii][jj]==0:
                    In=0 #ibid.
                else:
                    In=pdfXY[ii][jj]*np.log2(pdfXY[ii][jj]/(pdfX[ii]*pdfY[jj]))
                I=I+In
        #print("Mutual info is equal to", I)
        return I
    
    
    def _t_offset(self,x,y,steps):
    #Function for creating time offset btw. x and y
    #Positive offset means that the x precedes y
    #steps is the number of steps to be offset
        lx=len(x)
        if steps>=0:
            off_x=x[0:lx-steps]
            off_y=y[steps:]
        if steps<0:
            off_x=x[-steps:]
            off_y=y[0:lx+steps] 
        #print("x precedes y by",steps,"time steps")
        return off_x, off_y, steps

    def _remnans(self,x,y,prnt=None):
    #Takes in x and y
    #Removes the "nan" values from y and returns two arrays, where the nans have been removed from 
    #both x and y so that the time stamps stay
    #in on-to-one correspondence. Also prints the number of removed nans.
        nx=[]
        ny=[]
        for ii in range(len(y)):
            if am.all_nans(y[ii]):
                continue
            else:
                ny.append(np.float64(y[ii]))
                nx.append(np.float64(x[ii]))
                
        if prnt is not None:    
            print("Removed",len(y)-len(ny),"nans \n")
        return np.asarray(nx),np.asarray(ny)
    
    def _remnans_n(self,*args):
    #Remove Nans from an arbitray number of input arrays
        arrs=am._zero_array(args)
        new_arrs=np.zeros([0,len(args)])
        #Populate arrs
        for jj in range(len(args)):
            arrs[:,jj]=args[jj]
        #Go through all rows to check for Nans and remove rows with all Nans
        for ii in range(len(arrs[:,0])):
            if am.contains_nan(arrs[ii,:]):
                continue
            else:
                con_arr=np.array(arrs[ii,:],dtype=np.float64,ndmin=2)
                new_arrs=np.concatenate((new_arrs,con_arr),axis=0)
        new_arrs=np.asarray(new_arrs,dtype=np.float64)
        print("Removed %i nans \n" %(len(arrs[:,0])-len(new_arrs[:,0])))
        return new_arrs
            
    
    def _rembig(self,arr1,arr2,big):
    #Removes numbers numbers from an array and also, concatenates another array,
    #when certain numbers are used to denote missing values. TIme stamps stay untouched, i.e. "aligned"
    #It is assumed that arr2 contains the values to be removed
    #The removal is done based on absolute value.
        narr1=[]
        narr2=[]
        for ii in range(len(arr2)):
            if abs(arr2[ii])>=abs(big):
                continue
            else:
                narr1.append(np.float64(arr1[ii]))
                narr2.append(np.float64(arr2[ii]))
        print("Removed",len(arr2)-len(narr2),"'big' values")
        return np.asarray(narr1),np.asarray(narr2)
    
    def _big2nans(self,arr1,big):
    #Converts in absolute terms "big" values to Nans
        if big<0:
            print("Big must be positive, since the absolute values in the array are evaluated")
        small=arr1
        larr1=len(arr1)
        for jj in range(larr1):
            if am.all_nans(small[jj]):
                continue
            if abs(small[jj])>=big:
                small[jj]=np.nan
        return small
    
    def _tremnans(self,ulf,flx,toff):
    #The function creates the desired time step offset in the x and y data
    #and then removes the nans values from x and y data.
    #Returns the new x and y and the number of time steps for the offset
        npwr,nflx,t=self._t_offset(ulf,flx,toff)
        npwr1,nflx1=self._remnans(npwr,nflx)
        nflx1,npwr1=self._remnans(nflx1,npwr1)
        return npwr1,nflx1,t
    
    def _mi_t_offset(self,x,y,n_steps,tstep,ttl=None,plot=None,save=None,ret_vec=None):
    #Calculate the mutual info for the desired time offsets
        steps=np.linspace(-n_steps,n_steps,2*n_steps+1,dtype=np.int64)
        mi=np.empty(len(steps)) #Array to store mutual info
        corrs=np.empty([len(steps),2])
        toffs=np.empty(len(steps)) #Array to store time offsets
        #Lists for calculating the data vectors without Nans
        lst_nx=[]
        lst_ny=[]
        
        #Calculate mutual information and correlation
        for jj in range(len(steps)):
            nx,ny,tt=self._tremnans(x,y,steps[jj])
            lst_nx.append(nx)
            lst_ny.append(ny)
            #minflux=min(minflux,min(ny)) #Can be used, if minflux and pinpwr need to be checked
            #minpwr=min(minpwr,min(nx))
            mi[jj]=self._mutual_info(nx,ny)
            corrs[jj]=stats.pearsonr(nx,ny)
            toffs[jj]=tt*tstep
            #print(mi1)
            
        #Calculate noise
        noise=[]
        noise_sig=[]
        noise_var=[]
        runs=200
        for ii in range(len(steps)):
            #nn,ns,nt=self._tremnans(x,y,steps[ii])
            nn,ns,nv=self._mi_shuffle(lst_nx[ii],lst_ny[ii],runs)
            noise.append(nn)
            noise_sig.append(ns)
            noise_var.append(nv)    
        noise=np.asarray(noise)
        noise_sig=np.asarray(noise_sig)
        noise_var=np.asarray(noise_var)
        nsig=3 #Number of standard deviations
        
        if plot is not None:
            #plot results
            
            #plt.plot(toffs,corrs[:,0],"ko",label=str("Pearson correlation of original data"))
            #am.annot(toffs,corrs[:,0],corrs[:,1],"p") #Annotate p values to the correlations 
            am.plot_w_noise(toffs,mi,noise,noise_sig,nsig,"time offset","MI","data","noise",ttl)
            
        if save is not None:
            plt.savefig(ttl+".pdf",format="pdf")
        
        plt.show()
            
        #Return time offset and mutual info, and optionally the data vectors 
        if ret_vec==None:
            return toffs,mi
        else:
            return toffs,mi,nx,ny
        
    def _mi_w_noise(self,x,y):
        # mi=np.empty(len(steps)) #Array to store mutual info
        # corrs=np.empty([len(steps),2])
        # toffs=np.empty(len(steps)) #Array to store time offsets
        # #Lists for calculating the data vectors without Nans
        # lst_nx=[]
        # lst_ny=[]
        
        #Calculate mutual information and correlation
        #for jj in range(len(steps)):
        # nx,ny,tt=self._tremnans(x,y,steps[jj])
        # lst_nx.append(nx)
        # lst_ny.append(ny)
        #minflux=min(minflux,min(ny)) #Can be used, if minflux and pinpwr need to be checked
        #minpwr=min(minpwr,min(nx))
        mi=self._mutual_info(x,y)
        corr=stats.pearsonr(x,y)
        #toffs[jj]=tt*tstep
            #print(mi1)
            
        #Calculate noise
        runs=200
        #for ii in range(len(steps)):
        #nn,ns,nt=self._tremnans(x,y,steps[ii])
        noise,noise_sig,noise_var=self._mi_shuffle(x,y,runs)
        # noise.append(nn)
        # noise_sig.append(ns)
        # noise_var.append(nv)    
       
        # noise=np.asarray(noise)
        # noise_sig=np.asarray(noise_sig)
        # noise_var=np.asarray(noise_var)        
            
        #Return results
        return mi,noise,noise_sig,noise_var,corr
    
    def _mi_cond_binary(self,x,y,z,z_limit):
    #Evaluates the conditional mutual information I(x,y|z) by evaluating
    #the mutual information I(x,y) given z, weighted by the probability of z.
    #z is divided into two categories, given by z_limit
        n_z=len(z)
        #I=np.zeros(2)
        x=np.asarray(x)
        y=np.asarray(y)
        z=np.asarray(z)
        
        p_z_low=sum(1*(z_limit>=z))/n_z
        below=sum(1*(z_limit>=z))
        #above=n_z-below
        print("%i values of z are at or below the limit %f" %(below,z_limit))
        #print("%i values of z are above the limit %f" %(above,z_limit))
        x_z_low=x[z_limit>=z]
        y_z_low=y[z_limit>=z]
        #Check that y_z_low and x_z_low are not empty
        if len(x_z_low)<1 or len(y_z_low)<1:
            print("No z values below %f" %z_limit)
        else:
            #Evaluate mutual information and noise with error
            mi=p_z_low*self._mutual_info(x_z_low,y_z_low)
            noise,noise_sig,noise_var=self._mi_wei_shuffle(x_z_low,y_z_low,p_z_low,nruns=200)
            print('V%i; =mi=%f; noise=%f, sigma=%f' %(z_limit,mi,noise,noise_sig))
            print('%i & %.3f,%.3f $\\pm$ %.3f \\\\' %(z_limit,mi,noise,noise_sig))
            return mi,noise,noise_sig
    
        
        # p_z_high=1-p_z_low
        # x_z_high=x[z_limit<z]
        # y_z_high=y[z_limit<z]
        # #Check that y_z_high and x_z_high are not empty
        # if len(x_z_high)<1 or len(y_z_high)<1:
        #     print("No z values above %f" %z_limit)
        # else:
        #     I[1]=p_z_high*self._mutual_info(x_z_high,y_z_high)
        
        # print("Conditional mutual information is %f \n" %sum(I))
        # return sum(I)
    
    def _mi_cond_corridor(self,x,y,z,z_low,z_high):
    #Evaluates the conditional mutual information I(x,y|z) by evaluating
    #the mutual information I(x,y) given z, weighted by the probability of z.
    #z must be between z_low and z_high
        n_z=len(z)
        I=np.zeros(2)
        x=np.asarray(x)
        y=np.asarray(y)
        z=np.asarray(z)
        
        #Identify correct values of z
        z_c=1*(z>=z_low)*1*(z<=z_high)
        #z_c[z_c==2]=1

        p_z_in=sum(z_c)/n_z
        print("%i values of z are below the limit %f" %(sum(1*(z_low>=z)),z_low))
        print("%i values of z are above the limit %f" %(n_z-sum(1*(z_high>=z)),z_high))
        print("%i values of z are in the desired range" %sum(z_c))
        print("z has %i values in total" %n_z)
        x_z_in=x[z_c==1]
        y_z_in=y[z_c==1]
        #Check that y_z_in and x_z_in are not empty
        if len(x_z_in)<1 or len(y_z_in)<1:
            print("No solar wind btw. %f and %f" %(z_low,z_high))
        else:
            I[0]=p_z_in*self._mutual_info(x_z_in,y_z_in)
        
        p_z_out=1-p_z_in
        x_z_out=x[z_c==0]
        y_z_out=y[z_c==0]
        #Check that y_z_out and x_z_out are not empty
        if len(x_z_out)<1 or len(y_z_out)<1:
            print("No solar wind outside %f-%f" %(z_low,z_high))
        else:
            I[1]=p_z_out*self._mutual_info(x_z_out,y_z_out)
        
        print("The conditional mutual information is %f \n" %sum(I))
        return sum(I)
    
    def _mi_cond(self,x,y,z,pdf_xyz=None):
    #Evaluates the conditional mutual information I(x,y|z)
    #If the three-variable pdf is pre.calculkated, it can be given as input.
    #One can also declare that x, y, and z are 
        
        if pdf_xyz is not None:
            pdf_xyz=pdf_xyz
                 
        #Alternative and explicit, but slower way, to calculate pdf_xz and pdf_yz
        #pdf_xz=self._joint_pdf(x,z)
        #pdf_yz=self._joint_pdf(y,z)
        #pdf_z=self._create_pdf(z)
        else:
            arr=self._remnans_n(x,y,z)
            x,y,z=arr[:,0],arr[:,1],arr[:,2]
            pdf_xyz=self._joint3_pdf(x,y,z)      
        #Evaluete marginal pdfs
        pdf_z=np.sum(np.sum(pdf_xyz,axis=0),axis=0)
        pdf_xz=np.sum(pdf_xyz,axis=1)
        pdf_yz=np.sum(pdf_xyz,axis=0)
            
        #Evaluete conndtional mutual information
        I=0
        #Create 1D vectors for summing
        #pdf_xz=np.concatenate(pdf_xz)
        #pdf_yz=np.concatenate(pdf_yz)
        for ii in range(len(pdf_xz)):
            for jj in range(len(pdf_yz)):
                for kk in range(len(pdf_z)):
                #Handle division by zero and other exceptions
                #Since pdfxyz=0 if pdfz=0, we check only the values of pdfxyz
                    if (pdf_xyz[ii][jj][kk]>0 and pdf_xz[ii][kk]*pdf_yz[jj][kk]==0):
                        I=100 #This is a convention, see "Elements of Infomation Theory" OR SET TO SOME LARGE FINITE NUMBER
                        print("Mutual information has been set to 100. Check calculation")
                        return I
                    elif (pdf_xyz[ii][jj][kk]==0 and pdf_xz[ii][kk]*pdf_yz[jj][kk]==0):
                        In=0 #ibid.
                    elif (pdf_xz[ii][kk]*pdf_yz[jj][kk]!=0 and pdf_xyz[ii][jj][kk]==0):
                        In=0 #ibid.            
                    else:
                        In=pdf_xyz[ii][jj][kk]*np.log2(pdf_z[kk]*pdf_xyz[ii][jj][kk]/(pdf_xz[ii][kk]*pdf_yz[jj][kk]))
                    I=I+In
        #print("Mutual info is equal to", I)
        return I
        
    
    def _mi_shuffle(self,data1,data2,nruns=200):
    #To calculate the noise level using a shuffle test
        arrmi=np.empty(nruns)
        test=My_Mi('dummy1',1,data1,'dummy2',1,data2)
        for ii in range(nruns):    
            dn1=am.shuffler(test.x)     
            arrmi[ii]=self._mutual_info(dn1,data2)       
        #calculate average MI and variance for given correlatiohn
        avgmi=np.mean(arrmi,dtype=np.float64)
        varmi=np.var(arrmi,dtype=np.float64)
        sigmi=np.sqrt(varmi)
   
        return avgmi,sigmi,varmi
    
    def _mi_wei_shuffle(self,data1,data2,weight,nruns=200):
    #To calculate the noise level using a shuffle test with 
    #the mi weighted, e.f. for evaluating the noise of 
    #conditional mutual information
        arrmi=np.empty(nruns)
        test=My_Mi('dummy1',1,data1,'dummy2',1,data2)
        for ii in range(nruns):    
            dn1=am.shuffler(test.x)     
            arrmi[ii]=weight*self._mutual_info(dn1,data2)       
        #calculate average MI and variance
        avgmi=np.mean(arrmi,dtype=np.float64)
        varmi=np.var(arrmi,dtype=np.float64)
        sigmi=np.sqrt(varmi)
        
        return avgmi,sigmi,varmi
    
    def _cont_entropy(self,x):
    #The function calculates the entropy of a continuous variable
    #while taking into account the effect of bin width.
    #The formula is introduced by Jaynes in a 1962 lecture note from
    #Brandeis university, edited by K. W. Ford
    #The term log N is omitted.
        xmax=max(x)
        xmin=min(x)
        bins=self._n_bins(x)
        bin_width=(xmax-xmin)/bins
        pdf=self._create_pdf(x)
        pdf_0=self._if_null_probability(pdf)
        
        entropy_1=-1*pdf_0.dot(np.log2(pdf_0))
        entropy_2=np.log2(bin_width) 
        #entropy_2=np.log2(xmax-xmin)
        entropy=entropy_1+entropy_2
        
        return entropy
    
    def _disc_entropy(self,x):
    #The function evaluates the entropy of a discrete variable x
        pdf=self._create_pdf(x)
        pdf=self._if_null_probability(pdf)
        entropy=-1*(pdf.dot(np.log2(pdf)))
        #import pdb; pdb.set_trace()
        #print("Entropy is equal to", H)
        return entropy
    
    def _disc_jointentropy(self,x,y):
    #The function evaluates H(X,Y) for discrete variables X and Y
        pdfX=self._create_pdf(x)
        pdfY=self._create_pdf(y)
        pdfXY=self._joint_pdf(x,y)
    
        H=0
        #import pdb; pdb.set_trace()
        for ii in range(len(pdfX)):
            for jj in range(len(pdfY)):
                #skip any zero probabilities
                if pdfXY[ii][jj]==0:
                    continue
                summa=(pdfXY[ii][jj]*np.log2(pdfXY[ii][jj]))
                H=H-summa
        return H
    
    #NOT WORKING PROPERLY YET (DOESNT EVEN WORK FOR HIGHLY CORREATED GRVs
    #OR WITH A HIGH VARIANCE
    def _cont_jointentropy(self,x,y):
    #The function evaluates H(X,Y) for continuous variables X and Y
        pdfX=self._create_pdf(x)
        pdfY=self._create_pdf(y)
        pdfXY=self._joint_pdf(x,y)
        
        xmax=max(x)
        xmin=min(x)
        ymax=max(y)
        ymin=min(y)
        binsx=self._n_bins(x)
        binsy=self._n_bins(y)
        bin_width=(xmax-xmin)*(ymax-ymin)/(binsx*binsy)
    
        H=0
        #import pdb; pdb.set_trace()
        for ii in range(len(pdfX)):
            for jj in range(len(pdfY)):
                #skip any zero probabilities
                if pdfXY[ii][jj]==0:
                    continue
                summa=pdfXY[ii][jj]*np.log2(pdfXY[ii][jj]/bin_width) #????
                H=H-summa
        return H
        
    
    def _if_null_probability(self,x):
    #Convert zero probabilities to 1 so that they yield zero conttributio
    #to entropy
        nulls=1*x==0 #List zero probabilites
        x=x+nulls #Assign "1" instead of zero probability
        
        return x
