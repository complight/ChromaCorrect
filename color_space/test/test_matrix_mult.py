from email.mime import image
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

lms_matrix = torch.tensor([[8.5716e+00, 3.3698e+01, 2.0248e+00],
        [1.8377e+00, 3.4847e+01, 2.8624e+00],
        [3.3089e-08, 7.2596e-02, 3.1166e+01]]).float()
# lms_matrix2 = torch.tensor([[8.5716e+00, 1.8377e+00, 3.3089e-08],
#         [3.3698e+01, 3.4847e+01, 7.2596e-02],
#         [2.0248e+00, 2.8624e+00, 3.1166e+01]]).float()



print(lms_matrix)

directory = os.path.abspath(os.getcwd())
rgb_image_tensor = read_image(directory + '/data/parrot.png')

r = (rgb_image_tensor[0][0][0]).item()
g = (rgb_image_tensor[1][0][0]).item()
b = (rgb_image_tensor[2][0][0]).item()
print(str(r) + "  " + str(g) + "  "+ str(b))

rgb_tensor = torch.tensor([[r],[g],[b]]).float()/255

color = sample_under_display()
lms_image = color.rgb_to_lms(rgb_image_tensor)
print(lms_image[0][0][0])
print(lms_image[1][0][0])
print(lms_image[2][0][0]) 

print(torch.matmul(lms_matrix,rgb_tensor))
