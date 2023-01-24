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

L_n,M_n,S_n = perceptual_color.initialise_cones_normalised()

#reading the spectrum file (.csv)
directory = os.path.abspath(os.getcwd())
spectrum_file = directory + '/data/light_spectrum_sample.csv'
dataset = np.genfromtxt(spectrum_file,delimiter=',')
dataset = torch.from_numpy(dataset)

#instantiate display_spectrum 
R_spectrum_n = sample_under_display.initialise_display_spectrum_normalised(dataset)
L_resp_to_Red = sample_under_display.cone_response_to_spectrum(L_n,R_spectrum_n)
print('L cone response to display spectrum - Red : ' + str(L_resp_to_Red))
