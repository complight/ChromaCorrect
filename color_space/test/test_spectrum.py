import sys
import odak
import torch
import numpy as np
import matplotlib.pyplot as plt

red_data = np.swapaxes(np.genfromtxt('../backlight/red_spectrum.csv',delimiter=','),0,1)
green_data = np.swapaxes(np.genfromtxt('../backlight/green_spectrum.csv',delimiter=','),0,1)
blue_data = np.swapaxes(np.genfromtxt('../backlight/blue_spectrum.csv',delimiter=','),0,1)
wavelength = np.linspace(400, 700, num=301)
red_spectrum = torch.from_numpy(np.interp(wavelength, red_data[0], red_data[1])).unsqueeze(1)
green_spectrum = torch.from_numpy(np.interp(wavelength, green_data[0], green_data[1])).unsqueeze(1)
blue_spectrum = torch.from_numpy(np.interp(wavelength, blue_data[0], blue_data[1])).unsqueeze(1)
wavelength = torch.from_numpy(wavelength).unsqueeze(1) / 550.

curve = odak.learn.tools.multi_layer_perceptron(n_hidden=64).to(torch.device('cpu')) # device
curve.fit(wavelength.float(), red_spectrum.float(), epochs=1000, learning_rate=5e-4)
red_estimate = torch.zeros_like(red_spectrum)
for i in range(red_estimate.shape[0]):
    red_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
curve.fit(wavelength.float(), green_spectrum.float(), epochs=1000, learning_rate=5e-4)
green_estimate = torch.zeros_like(green_spectrum)
for i in range(green_estimate.shape[0]):
    green_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
curve.fit(wavelength.float(), blue_spectrum.float(), epochs=1000, learning_rate=5e-4)
blue_estimate = torch.zeros_like(blue_spectrum)
for i in range(blue_estimate.shape[0]):
    blue_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
    
primary_wavelength = torch.linspace(400,700,301)
red_spectrum_fit = torch.cat((primary_wavelength.unsqueeze(1), red_estimate),1)
green_spectrum_fit = torch.cat((primary_wavelength.unsqueeze(1), green_estimate),1)
blue_spectrum_fit = torch.cat((primary_wavelength.unsqueeze(1), blue_estimate),1)
red_spectrum_fit[:,1] *= (red_spectrum_fit[:,1] > 0)
green_spectrum_fit[:,1] *= (green_spectrum_fit[:,1] > 0)
blue_spectrum_fit[:,1] *= (blue_spectrum_fit[:,1] > 0)

fig, ax = plt.subplots(1)
wavelength = wavelength.detach().cpu().numpy()
ax.plot(red_spectrum_fit[:,0].detach().cpu().numpy(), red_spectrum_fit[:,1].detach().cpu().numpy(), color='r')
ax.plot(green_spectrum_fit[:,0].detach().cpu().numpy(), green_spectrum_fit[:,1].detach().cpu().numpy(), color='g')
ax.plot(blue_spectrum_fit[:,0].detach().cpu().numpy(), blue_spectrum_fit[:,1].detach().cpu().numpy(), color='b')
ax.plot(wavelength*550., red_spectrum, color='black')
ax.plot(wavelength*550., green_spectrum, color='black')
ax.plot(wavelength*550., blue_spectrum, color='black')

plt.show()