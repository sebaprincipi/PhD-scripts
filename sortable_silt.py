# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:44:02 2023

@author: Sebastian Principi - sebaprincipi@gmail.com
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

def read_cilas_data(csv):
    # Read the CSV file containing the granulometry data
    data = pd.read_csv(csv,encoding='latin-1',sep=";")
    
    
    # Calculate the sortable silt fraction (10–63 μm silt fraction)
    sortable_silt_data = data[(data['Silt'] >= 10) & (data['Silt'] <= 63)]
    
    # Calculate the sortable silt mean size
    sortable_silt_data['Mean Size'] = (sortable_silt_data['Silt'] + sortable_silt_data['Clay']) / 2
    
    # Calculate the sortable silt percentage
    sortable_silt_data['Sortable Silt Percentage'] = (sortable_silt_data['Silt'] / (sortable_silt_data['Sand'] + sortable_silt_data['Silt'] + sortable_silt_data['Clay'])) * 100
    
    name=csv[:-4]
    
    return sortable_silt_data,name

def SS_perc_mean(sortable_silt_data,name):
    
    #Graph 1: Sortable silt mean vs sortable silt percentage
    
    plt.subplots(figsize=(8, 12))
    
    # Perform linear regression
    X = sortable_silt_data['Mean Size'].values.reshape(-1, 1)
    y = sortable_silt_data['Sortable Silt Percentage'].values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    
    plt.subplot(3, 1, 1)
    # Calculate R-squared value
    r_squared = r2_score(y, y_pred)
    
    # Plotting the correlation graph with linear regression line and R-squared text
    plt.scatter(sortable_silt_data['Mean Size'], sortable_silt_data['Sortable Silt Percentage'], marker='o', color='b')
    plt.plot(X, y_pred, color='r')
    
    # Set the x-axis label
    plt.xlabel('Sortable Silt Mean Size (μm)')
    # Set the y-axis label
    plt.ylabel('Sortable Silt Percentage (%)')
    

    # Display the R-squared value as text on the graph
    plt.text(0.1, 0.9, f'R-squared = {r_squared:.2f}', transform=plt.gca().transAxes)

        
#%%

    #Graph 2: Core length vs Sortable silt mean
    plt.subplot(3, 1, 2)
    
    # Core depth
    plt.plot( sortable_silt_data['Depth'].values, sortable_silt_data['Mean Size'].values, marker='o', color='b')

    # Set the x-axis label
    plt.xlabel('Core length (cm)')
    # Set the y-axis label
    plt.ylabel('Sortable Silt Mean Size (μm)')
    
            
#%%

    
    #Graph 3: Core length vs velocity U (cm/seg)
    
    plt.subplot(3, 1, 3)
    
    # Velocity calculation. From McCave 2017 https://www.sciencedirect.com/science/article/pii/S0967063717300754
    U=sortable_silt_data['Mean Size'].values*1.31 -17.18
    plt.plot( sortable_silt_data['Depth'].values, U, marker='o', color='b')
    # Set the x-axis label
    plt.xlabel('Core length (cm)')
    # Set the y-axis label
    plt.ylabel('U (cm/seg)')
    
    plt.tight_layout()
    plt.savefig(name+'.png')
    plt.show()
    

#Example    
csv=r'D:\Onedrive_sebastian\OneDrive - gl.fcen.uba.ar\Doc\4.Base de datos\7.Corings\Cores_GEO4_5\Au_Geo05_GC135.csv'
SS_data,name=read_cilas_data(csv)
SS_perc_mean(SS_data,name)