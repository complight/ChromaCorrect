import torch
from torch import fft
import math
import numpy as np
from .zernike_polynomial_generator import zernike_polynomial_generator


class optical_aberration_generator():

    """
    A class for modelling optical aberrations using Point Spread Functions (PSFs)
    """

    def __init__(self, kernel_size, coefficients, nm_list,
                n_range=4, eye_pupil_diameter=6,
                psf_size_degrees=0.45,
                spherical_power=0,
                cylindrical_power=0,
                axis_angle=0
                ):
        self.kernel_size = kernel_size
        self.coefficients = coefficients
        self.nm_list = nm_list
        self.n_range = n_range
        self.zernike_generator = zernike_polynomial_generator()
        self.eye_pupil_diameter = eye_pupil_diameter
        self.psf_size_degrees = psf_size_degrees
        self.spherical_power=spherical_power
        self.cylindrical_power=cylindrical_power
        self.axis_angle=axis_angle
        self.set_sphero_cylindrical_coefficients()

    def calculate_pupil_samples(self, light_wavelength):
        degrees = math.pi/180
        return 10**6 * self.psf_size_degrees * self.eye_pupil_diameter * degrees / light_wavelength

    def equivalent_defocus_in_diopters(self):
        defocus_dioptre = 16 * math.sqrt(3) * self.eye_pupil_diameter**-2 * self.coefficients[4]
        print('defocus in dioptres : ', defocus_dioptre)

    def set_defocus_coefficient_from_diopters(self, defocus):
        self.coefficients[4] = (defocus * self.eye_pupil_diameter**2) / (16*math.sqrt(3))
        print('defocus coefficient : ', self.coefficients[4])

    def set_sphero_cylindrical_coefficients(self):
        self.coefficients[3] = (self.cylindrical_power * math.sin(2 * self.axis_angle) * (self.eye_pupil_diameter**2)) / (8 * math.sqrt(6))
        self.coefficients[4] = (self.spherical_power * (self.eye_pupil_diameter**2))/(16 * math.sqrt(3))
        self.coefficients[5] = (self.cylindrical_power * math.cos(2 * self.axis_angle) * (self.eye_pupil_diameter**2)) / (8 * math.sqrt(6))

        print('oblique astigmatism coefficient from diopters : ', self.coefficients[3])
        print('defocus coefficient from diopters : ', self.coefficients[4])
        print('vertical astigmatism coefficient from diopters : ', self.coefficients[5])

    def weighted_zernike_polynomials(self):
        """
        Fetches a list of Zernike polynomials according to order n

        Parameters
        ----------

        Returns
        -------
        zernike_polynomials        : list
                                     Compilation of Zernike polynomials
        """
        zernike_polynomials = []
        for (n, m), c in zip(self.nm_list, self.coefficients):
            _, _, Z_polar = self.zernike_generator.generate_zernike_polar(n, m, self.kernel_size)
            zernike_polynomials.append(c * Z_polar)
        return zernike_polynomials


    def generate_optical_aberration(self):
        """
        Model a optical aberrations as a weighted sum of Zernike polynomials

        Parameters
        ----------

        Returns
        -------
        optical_aberration        :  torch.tensor
                                     Model of optical aberrations
        """
        pupil_radius = math.ceil(self.pupil_radius)
        zernike_polynomials = self.weighted_zernike_polynomials()
        optical_aberration = torch.zeros(self.kernel_size)
        optical_aberration = torch.FloatTensor(sum(zernike_polynomials))
        return optical_aberration


    def generate_pupil_mask(self, light_wavelength):
        """
        Create a mask of 1s inside the pupil radius and 0s outside

        Parameters
        ----------

        Returns
        -------
        pupil_mask             :     torch.tensor
                                     Mask that is 1s inside pupil radius
        """
        pupil_samples = self.calculate_pupil_samples(light_wavelength)
        self.pupil_samples = pupil_samples
        self.pupil_radius = self.pupil_samples / 2
        pupil_radius = math.ceil(self.pupil_radius)
        x = torch.linspace(-1, 1, 2*pupil_radius)
        y = torch.linspace(-1, 1, 2*pupil_radius)
        Y, X = torch.meshgrid(x, y, indexing='ij')
        rhos, phis = torch.sqrt(X**2 + Y**2), torch.atan2(Y, X)
        M = torch.cos(phis)**2 + torch.sin(phis)**2
        M[rhos>1] = 0
        pupil_mask = torch.zeros(self.kernel_size)
        pupil_mask[self.kernel_size[0]//2-pupil_radius+1:self.kernel_size[0]//2+pupil_radius+1, self.kernel_size[1]//2-pupil_radius+1:self.kernel_size[1]//2+pupil_radius+1] = M
        return pupil_mask


    def generalized_pupil_function(self, light_lambda=None):
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
        pupil_mask = self.generate_pupil_mask(light_wavelength)
        optical_aberration = self.generate_optical_aberration()
        gpf = pupil_mask * torch.exp(((1j * 2 * torch.pi * 10**3) / light_wavelength) * optical_aberration)
        return gpf


    def generate_psf_for_wavelengths(self, wavelengths):
        """
        Calculates the point spread function for a list of wavelengths.

        Parameters
        ----------
        wavelengths      :           float
                                     Wavelength of light.
        Returns
        -------
        psf             :            torch.tensor
                                     Point spread function kernel.
        """
        channel_psfs = []
        for wavelength in wavelengths:
            channel_gpf = self.generalized_pupil_function(wavelength)
            channel_psf = fft.ifftshift(fft.fft2(fft.fftshift(channel_gpf)))
            channel_psf = (torch.abs(channel_psf))**2
            channel_psf = channel_psf/channel_psf.sum()
            channel_psfs.append(channel_psf)
        psf = torch.stack([channel_psf for channel_psf in channel_psfs], dim=2)
        return psf

    def generate_psf_for_wavelength(self, wavelength):
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
        gpf = self.generalized_pupil_function(wavelength)
        psf = fft.ifftshift(fft.fft2(fft.fftshift(gpf)))
        psf = (torch.abs(psf))**2
        psf = psf/psf.sum()
        return psf
