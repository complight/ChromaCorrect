from .zernike_polynomial_generator import zernike_polynomial_generator
from .optical_aberration_generator import optical_aberration_generator
from color_space.display import sample_under_display
import torch
import odak

def create_gaussian_kernel(target_image, nsigma=[2, 2]):
    import torch
    import odak
    kernel = torch.zeros_like(target_image)
    for i in range(3):
       kernel[:, :, i] = odak.learn.tools.generate_2d_gaussian(
                                                               kernel_length=[
                                                                              kernel.shape[0],
                                                                              kernel.shape[1]
                                                                             ],
                                                               nsigma=nsigma
                                                              )
    return kernel


def use_eye_power(switch):
    eye_power = None
    if switch:
        eye_power = input('Default eye power is set to 0.5, enter a value if you want to change it: ')
        eye_power = 0.5 if eye_power == '' else float(eye_power)
    return eye_power


def create_optical_aberration_kernel(
                                     target_image,
                                     zernike_coefficients = [
                                                              0.0,   # Piston
                                                              0.0,   # Y-Tilt
                                                              0.0,   # X-Tilt
                                                              0.0,   # Oblique Astigmatism
                                                              0.0,   # Defocus
                                                              0.0,   # Vertical Astigmatism
                                                              0.0,   # Vertical Trefoil
                                                              0.0,   # Vertical Coma
                                                              0.0    # Horizontal Coma
                                                            ],
                                     nm_list = [
                                                [0, 0],     # Piston
                                                [1, -1],    # Y-Tilt
                                                [1, 1],     # X-Tilt
                                                [2, -2],    # Oblique Astigmatism
                                                [2, 0],     # Defocus
                                                [2, 2],     # Vertical Astigmatism
                                                [3, -3],    # Vertical Trefoil
                                                [3, -1],    # Vertical Coma
                                                [3, 1],     # Horizontal Coma
                                               ],
                                     spherical_power=0,
                                     cylindrical_power=0,
                                     axis_angle=0,
                                     psf_size_degrees=0.45,
                                     eye_pupil_diameter=6,
                                     wavelengths=[620, 555, 470]
                                    ):
        kernel_size = [target_image.shape[0], target_image.shape[1]]
        eye_power = use_eye_power(False)
        optical_aberration_gen = optical_aberration_generator(
                                                              kernel_size,
                                                              zernike_coefficients,
                                                              nm_list,
                                                              spherical_power=spherical_power,
                                                              cylindrical_power=cylindrical_power,
                                                              axis_angle=axis_angle,
                                                              eye_pupil_diameter=eye_pupil_diameter, # in mm
                                                              psf_size_degrees=psf_size_degrees
                                                              )
        psf = optical_aberration_gen.generate_psf_for_wavelengths(wavelengths)
        return psf

def create_optical_aberation_kernel_4D(
                                    kernel_size,
                                    wavelengths,
                                    intensities,
                                    color_spectrum,
                                    kernel_type,
                                    zernike_coefficients = [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    spherical_power=0,
                                    cylindrical_power=0,
                                    axis_angle=0,
                                    psf_size_degrees=0.45,
                                    eye_pupil_diameter=6,
                                    device=None
                                ):
        """
        Function to generate combined LMS response for given wavelengths and their corresponding intensities.

        Parameters
        ----------
        kernel_size           : list
                                Kernel size.
        wavelengths           : list
                                List of wavelengths. This list contains X number of wavelengths.
        intensities           : list
                                Normalized intensity for each provided wavelength. This list contains X number of intensities.
        color_spectrum        : class
                                Color spectrum class.
        kernel_type           : string
                                kernel type
        zernike_coefficients  : list
                                Zernike coefficients (Piston, Y-tilt, X-tilt, Oblique astigmatis, defocus and vertical astigamatism terms.
        device                : torch.device
                                If no device is presented, it will choose to use CPU.

        Returns
        -------
        lms_kernel            : torch.tensor
                                LMS response for the given wavelengths and intensities.
        """
        if isinstance(device, type(None)):
            device = torch.device('cpu')
        nm_list = [ [0, 0], [1, -1], [1, 1], [2, -2], [2, 0], [2, 2], [3, -3], [3, -1], [3, 1] ]
        optical_aberration = optical_aberration_generator(
                                                        kernel_size,
                                                        zernike_coefficients,
                                                        nm_list,
                                                        spherical_power=spherical_power,
                                                        cylindrical_power=cylindrical_power,
                                                        axis_angle=axis_angle,
                                                        eye_pupil_diameter=6,
                                                        psf_size_degrees=psf_size_degrees
                                                        )
        lms_kernel = torch.zeros(kernel_size[0],kernel_size[1], 3).to(device)
        for wavelength_id, wavelength in enumerate(wavelengths):
            intensity = intensities[wavelength_id]
            kernel = optical_aberration.generate_psf_for_wavelength(wavelength).to(device)
            weighted_kernel = intensity * kernel
            if kernel_type == 'LMS':
                lms_kernel += color_spectrum.convert_to_lms(weighted_kernel, wavelength, intensity)    
        return lms_kernel


def get_display_kernel_4D(
                        target_image,
                        kernel_type,
                        zernike_coefficients=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        spherical_power=0,
                        cylindrical_power=0,
                        axis_angle=0,
                        psf_size_degrees=0.45,
                        eye_pupil_diameter=6,
                        backlight = 'default',
                        device=None
                        ):
    """
    Function to use eye power.

    Parameters
    ----------
    target_image          : torch tensor
                            image
    kernel_type           : string
                            kernel type
    zernike_coefficients  : list
                            Zernike coefficients (Piston, Y-tilt, X-tilt, Oblique astigmatis, defocus and vertical astigamatism terms.
    device                : torch.device
                            If no device is presented, it will choose to use CPU.

    Returns
    -------
    kernel                : torch.tensor [4 dimensional]
                            Returns 4 dimensional vector which has LMS values for each color primaries.
    """
    if isinstance(device, type(None)):
        device = torch.device('cpu')
    color_spectrum = sample_under_display(device=device, read_spectrum='default')
    if backlight == 'default':
            red_spectrum, green_spectrum, blue_spectrum = color_spectrum.initialise_rgb_backlight_spectrum()
    else:
            red_spectrum, green_spectrum, blue_spectrum = color_spectrum.initialise_normalised_spectrum_primaries()

    entire_spectrum = [red_spectrum, green_spectrum, blue_spectrum]
    kernel = torch.zeros(3, target_image.shape[0], target_image.shape[1], target_image.shape[2]).to(device)
    for spectrum_id, spectrum in enumerate(entire_spectrum):
        print('4D LMS kernel is being created [for color primary {:.0f}]'.format(spectrum_id + 1))
        kernel[spectrum_id, :, :, :] = create_optical_aberation_kernel_4D(
                                                                    kernel_size=[
                                                                                target_image.shape[0],
                                                                                target_image.shape[1],
                                                                                ],
                                                                    wavelengths=torch.arange(400, 701),
                                                                    intensities=spectrum,
                                                                    zernike_coefficients=zernike_coefficients,
                                                                    spherical_power=spherical_power,
                                                                    cylindrical_power=cylindrical_power,
                                                                    axis_angle=axis_angle,
                                                                    psf_size_degrees=psf_size_degrees,
                                                                    eye_pupil_diameter=eye_pupil_diameter,
                                                                    color_spectrum=color_spectrum,
                                                                    kernel_type = kernel_type,
                                                                    device=device
                                                                ).to(device)

    return kernel
