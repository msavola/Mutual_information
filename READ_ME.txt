3.8.2020 by Mikko Savola

Notes on files main_mi.py and aux2_mi.py for calculating mutual information.

The file aux_mi.py contains auxiliary function for reading text files and plotting results.
The file main_mi.py contains the class definition for creating objects and calculating mutual information.

All estimates of entropy and mutual information are in bits.

The object instance has the following input parameters:
file1		is a string containing the file name of one data series
dates		is an array containing the time stamps of the data series
x 		contains one time series
file 2 		is a string containing the file name of the second data series
Lshells 	is a legacy variable (to be removed, when code will be updated)
y 		contains the other time series

An instance is created with the syntax:
obj=My_MI(file1,dates,x,file2,Lshells,y)

The following object functions can be used to evaluate entropy and mutual information:
obj._mutual_info(obj.x,obj.y) returns the mutual information of x and y. This estimate calculates discrete probability mass functions of x and y to estimate mutual information but x and y can be continuous.

obj._mi_t_offset(x,y,n_steps,tstep) calculates the mutual information of x and for desired time offset.n_Steps is the number of steps (+-) for calculating the offset and tstep is the size of one step (unit of time is implied). ttl can be given if a plot title is desired. Setting plot not equal to None plots the results with a noise level of +-3 sigma using a shuffle test of 200 shuffles. If save is set to other than None, the plot is saved in the working directory.

obj._mi_shuffle(x,y) calculates the average mutual information of x and y by doing a shuffle test 200 times and taking the average. This can be used for estimating the error in mutual information.

obj._cont_entropy(x) evaluates the entropy of continuous variable x.

obj._disc_entropy(x) evaluates the entropy of discrete variable x.

obj._disc_jointentropy(x,y) evaluates the entropy of discrete variables x and y.