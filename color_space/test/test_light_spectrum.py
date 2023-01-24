import os
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys 
import odak
sys.path.append('/home/atlas/cll/learned_prescription/color_space/')
from display import sample_under_display

#reading the spectrum file (.csv)
# # directory = os.path.abspath(os.getcwd())
# # data_file = directory + '/data/light_spectrum_sample.csv'
# dataset = np.genfromtxt(data_file,delimiter=',')
# dataset = torch.from_numpy(dataset)

# creating spectrum object
display_space = sample_under_display()
monitor_spectrum = display_space.initialise_rgb_backlight_spectrum()
wavelength = [i + 400 for i in range(301)]
print(monitor_spectrum[0].shape)
plt.plot(wavelength, monitor_spectrum[2],linewidth = '7.0', color = 'b')
plt.xlabel('Wavelength')
plt.ylabel('Normalised Intensity')

plt.savefig('spectrum.png')


#Target data plotting
# dataset_target = dataset.swapaxes(0,1)
# plt.plot(dataset_target.detach().numpy()[0],dataset_target.detach().numpy()[1], color='black',linestyle = 'dashed',label = 'target')
# plt.show()

