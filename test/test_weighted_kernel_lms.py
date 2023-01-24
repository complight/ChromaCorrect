from optical_aberration.optical_aberration_generator import optical_aberration_generator
from optical_aberration.zernike_polynomial_generator import zernike_polynomial_generator
from color_space.display import sample_under_display
import torch
import odak
from torch import fft
import math
from odak.learn.tools import save_image
from odak.tools import check_directory

filename = './dataset/parrot.png'
target_image = odak.learn.tools.load_image(filename)[:,:,0:3] / 255.
zernike_generator = zernike_polynomial_generator()
output_directory = "./output_conver_test/"
check_directory(output_directory)

zernike_coefficients=[
    0.0,     # Piston
    0.0,     # Y-Tilt
    0.0,     # X-Tilt
    0.0,     # Oblique Astigmatism
    0.0,     # Defocus
    0.0,     # Vertical Astigmatism
    0.0,     # Vertical Trefoil
    0.0,     # Vertical Coma
    0.0      # Horizontal Coma
]

def weighted_zernike_polynomials():
    """
    Fetches a list of Zernike polynomials according to order n

    Parameters
    ----------

    Returns
    -------
    zernike_polynomials        : list
                                    Compilation of Zernike polynomials
    """

    kernel_size = [512,512]
    nm_list = [ [0, 0], [1, -1], [1, 1], [2, -2], [2, 0], [2, 2], [3, -3], [3, -1], [3, 1] ]
    zernike_polynomials = []
    for (n, m), c in zip(nm_list, zernike_coefficients):
        _, _, Z_polar = zernike_generator.generate_zernike_polar(n, m, kernel_size)
        zernike_polynomials.append(c * Z_polar)
    return zernike_polynomials

def generate_optical_aberration():
    """
    Model a optical aberrations as a weighted sum of Zernike polynomials

    Parameters
    ----------

    Returns
    -------
    optical_aberration        :  torch.tensor
                                    Model of optical aberrations
    """

    kernel_size = [512,512]
    zernike_polynomials = weighted_zernike_polynomials()
    optical_aberration = torch.zeros(kernel_size)
    optical_aberration = torch.FloatTensor(sum(zernike_polynomials))
    return optical_aberration

def calculate_pupil_samples(light_wavelength):
    psf_size_degrees=0.45
    eye_pupil_diameter=6
    degrees = math.pi/180
    return 10**6 * psf_size_degrees * eye_pupil_diameter * degrees / light_wavelength

def generate_pupil_mask(light_wavelength):
    """
    Create a mask of 1s inside the pupil radius and 0s outside

    Parameters
    ----------

    Returns
    -------
    pupil_mask             :     torch.tensor
                                    Mask that is 1s inside pupil radius
    """
    kernel_size = [512,512]
    pupil_samples = calculate_pupil_samples(light_wavelength)
    pupil_radius = pupil_samples / 2
    pupil_radius = math.ceil(pupil_radius)
    x = torch.linspace(-1, 1, 2*pupil_radius)
    y = torch.linspace(-1, 1, 2*pupil_radius)
    Y, X = torch.meshgrid(x, y, indexing='ij')
    rhos, phis = torch.sqrt(X**2 + Y**2), torch.atan2(Y, X)
    M = torch.cos(phis)**2 + torch.sin(phis)**2
    M[rhos>1] = 0
    pupil_mask = torch.zeros(kernel_size)
    pupil_mask[kernel_size[0]//2-pupil_radius+1:kernel_size[0]//2+pupil_radius+1, kernel_size[1]//2-pupil_radius+1:kernel_size[1]//2+pupil_radius+1] = M
    return pupil_mask

def generalized_pupil_function(light_lambda=None):
    """
    Calculates generalized pupil function using optical aberration and pupil mask

    Parameters
    ----------
    light_lambda    :            int
                                    Light wavelenth in nm
    Returns
    -------
    gpf             :            torch.tensor
                                    Generalized pupil function
    """
    light_wavelength = 620 if not light_lambda else light_lambda
    pupil_mask = generate_pupil_mask(light_wavelength)
    optical_aberration = generate_optical_aberration()
    gpf = pupil_mask * torch.exp(((1j * 2 * torch.pi * 10**3) / light_wavelength) * optical_aberration)
    return gpf

def generate_psf_for_wavelength( wavelength):
    """
    Calculates the point spread function for a given wavelength.

    Parameters
    ----------
    wavelength      :            float
                                    Wavelength of light.
    Returns
    -------
    psf             :            torch.tensor
                                    Point spread function kernel.
    """
    gpf = generalized_pupil_function(wavelength)
    psf = fft.ifftshift(fft.fft2(fft.fftshift(gpf)))
    psf = (torch.abs(psf))**2
    psf = psf/psf.sum()
    return psf

wavelengths=torch.arange(400, 701)
color_spectrum = sample_under_display(device='cpu', read_spectrum='default')
lms_kernel = torch.zeros(target_image.shape[0],target_image.shape[0], 3)

red_spectrum, green_spectrum, blue_spectrum = color_spectrum.initialise_normalised_spectrum_primaries()

spectrum = red_spectrum
for wavelength_id, wavelength in enumerate(wavelengths):
    intensity = spectrum[wavelength_id]
    kernel = generate_psf_for_wavelength(wavelength)
    weighted_kernel = intensity * kernel
    lms_kernel += color_spectrum.convert_to_lms(weighted_kernel, wavelength, intensity)    
print(lms_kernel[:,:,0].max())
print(lms_kernel[:,:,1].max())
print(lms_kernel[:,:,2].max())
save_image('{}/LMS_kernel_spec.png'.format(output_directory), ((lms_kernel - lms_kernel.min())/(lms_kernel.max()-lms_kernel.min()) * 255))

import matplotlib.pyplot as plt
plt.plot(spectrum)
plt.show()