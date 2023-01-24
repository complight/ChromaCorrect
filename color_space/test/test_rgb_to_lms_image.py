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

device_ = torch.device("cpu")

directory = os.path.abspath(os.getcwd())
rgb_image = read_image(directory + '/data/parrot.png') /255.0


#Testing RGB->LMS function and the constructor `lms_tensor`
#l_normalised, m_normalised, s_normalised = perceptual_color.initialise_cones_normalised()
color_display = sample_under_display(device=device_) 
print(color_display.device)      
lms_image = color_display.rgb_to_lms(rgb_image.to(device_)) #remove lms_matrix by passing the lms_tensor inside display.py

#Testing call function 
loss = color_display(rgb_image,rgb_image)
print(loss)
