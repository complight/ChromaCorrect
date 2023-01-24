import torch
import math


class zernike_polynomial_generator():

    """
    A class for computing Zernike polynomials in polar coordinates
    """

    def __init__(self):
        # rho is the normalized pupil radius between 0 to 1
        # phi is the azimuthal angle around the pupil between 0 to 2pi
        pass

    def zernike_normalization_factor(self, n, m):
        d_m0 = 1 if m == 0 else 0   # Kronecker delta function
        return math.sqrt( (2*(n+1))/(1+d_m0) )

    def zernike_radial_component(self, n, m, rho):
        """
        Computes the radial component R, of the Zernike polynomial

        Parameters
        ----------
        n                :  int
                            Order
        m                :  int
                            Frequency
        rho              :  torch.tensor
                            Normalized radial distance, in polar coordinates,
                            ranges from 0 to 1

        Returns
        -------
        R                : torch.tensor
                           Radial component of the Zernike polynomial
        """
        R = torch.zeros_like(rho)
        R[rho==1] = 1
        if (n-m) % 2 == 0:
            non_zero_indices = rho!=1
            for k in range((n-m)//2 + 1):
                nominator = ((-1)**k) * math.factorial(n-k)
                denominator = math.factorial(k) * math.factorial((n+m)//2 - k) * math.factorial((n-m)//2 - k)
                R[non_zero_indices] += (nominator / denominator) * (rho[non_zero_indices]**(n - 2*k))
        return R


    def zernike_polynomial(self, n, m, rho, phi):
        """
        Computes the Zernike polynomial for the given parameters

        Parameters
        ----------
        n                :  int
                            Order
        m                :  int
                            Frequency
        rho              :  torch.tensor
                            Normalized radial distance, in polar coordinates,
                            ranges from 0 to 1
        phi              :  torch.tensor
                            Azimuthal angle, in polar coordinates, ranges from
                            0 to 2pi

        Returns
        -------
        Z_poly           : torch.tensor
                           Zernike polynomial for the given parameters
        """
        R = self.zernike_radial_component(n, abs(m), rho)
        N = self.zernike_normalization_factor(n, m)
        if m < 0:
            m = abs(m)
            Z_poly = torch.sin(m * phi)
        else:
            Z_poly = torch.cos(m * phi)
        return N * R * Z_poly


    def generate_zernike_polar(self, n, m, resolution):
        """
        Computes the Zernike polynomial in polar coordinates

        Parameters
        ----------
        n                :  int
                            Order
        m                :  int
                            Frequency

        Returns
        -------
        Z_polar           : torch.tensor
                            Zernike polynomial for polar coordinates
        """
        x = torch.linspace(-1, 1, resolution[0])
        y = torch.linspace(-1, 1, resolution[1])
        Y, X = torch.meshgrid(x, y, indexing='ij')
        rhos, phis = torch.sqrt(X**2 + Y**2), torch.arctan2(Y, X)
        Z_polar = self.zernike_polynomial(n, m, rhos, phis)
        Z_polar[rhos>1] = 0
        return rhos, phis, Z_polar
