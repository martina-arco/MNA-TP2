
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cmath import exp, pi
import pandas as pd #To Convert your lists to pandas data frames convert your lists into pandas dataframes

from scipy import stats # For in-built method to get PCC

# data = {'list 1':[112.5412566015678, 98.47359142016575, 73.85519362521076,  72.09673828125],'list 2':[10,20,30,60]}

data = {'list 1':[112.5412566015678, 98.47359142016575, 73.85519362521076,  72.09673828125],'list 2':[10,20,30,60]}

df = pd.DataFrame(data, columns=['list 1','list 2'])

pearson_coef, p_value = stats.pearsonr(df["list 1"], df["list 2"]) #define the columns to perform calculations on
print("Pearson Correlation Coefficient: ", pearson_coef, "and a R al cuadrado of:", pearson_coef*pearson_coef) # Results