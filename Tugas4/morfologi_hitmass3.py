# importing required libraries 
import mahotas as mh 
import numpy as np 
from pylab import imshow, show 
	
# creating region 
# numpy.ndarray 
regions = np.zeros((10, 10), bool) 
	
# setting 1 value to the region 
regions[2:3, :3] = 1
regions[7:, 7:] = 1
	
# showing the image with interpolation = 'nearest' 
print("Image") 
imshow(regions, interpolation ='nearest') 
show() 

# template for hit miss 
template = np.array([ 
			[0, 1, 1], 
			[0, 1, 1], 
			[0, 1, 1]]) 

# hit miss transform 
img = mh.hitmiss(regions, template) 

# showing image 
print("Image after hit miss transform") 
imshow(img) 
show() 

