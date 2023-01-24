import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
import pandas as pd
import sys
sys.path.append('../')
from display.display import sample_under_display
from retina.retina import perceptual_color
def plot_SLM_data(csv_data):

    df = pd.read_csv(csv_data)

    S_wavelength = list(df['S-WaveL'])
    S_value = list(df['S-Value'])
    L_wavelength = list(df['L-WaveL'])
    L_value = list(df['L-Value'])
    M_wavelength = list(df['M-WaveL'])
    M_value = list(df['M-Value'])

    plt.plot(S_wavelength,S_value, color='black',linestyle = 'dashed', label='S')
    plt.plot(L_wavelength,L_value, color='black',linestyle = 'dashed', label ='M')
    plt.plot(M_wavelength,M_value, color='black',linestyle = 'dashed',label ='L')


directory = os.path.abspath(os.getcwd())
data_file_lms = directory + '/data/LMS.csv'
data_file_rgb = directory + '/data/led_backlight_spectrum.csv'

L,M,S = perceptual_color.initialise_cones()
L_n,M_n,S_n = perceptual_color.initialise_cones_normalised()

plot_SLM_data(data_file_lms)
perceptual_color.plot_response(L)
perceptual_color.plot_response(M)
perceptual_color.plot_response(S)
plt.show()

#Retrieving peak spectra points
print(perceptual_color.cone_response(567.5,L_n))
print(perceptual_color.cone_response(545,M_n))
print(perceptual_color.cone_response(447.5,S_n))
