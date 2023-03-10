import sys
import os
import odak
import torch
import argparse
from optimizers import learned_prescription_optimizer
from optical_aberration import create_optical_aberration_kernel, get_display_kernel_4D
from color_space.display import sample_under_display


__title__ = 'Prescription corrected image generator'


def main():
    no_of_iterations = 100
    learning_rate=0.01
    filename = './dataset/parrot.png'
    output_directory = './output'
    device_name = 'cpu'
    backlight = 'default'
    kernel_type = 'LMS'
    prescription = 'myopia'
    diopter = -2.00
    parser = argparse.ArgumentParser(description=__title__)
    parser.add_argument(
                        '--iterations',
                        type=int,
                        help='Number of optimization iterations/steps. Default is {}'.format(no_of_iterations)
                       )
    parser.add_argument(
                        '--learningrate',
                        type=float,
                        help='Learning rate used for the optimization. Default is {}'.format(learning_rate)
                       )
    parser.add_argument(
                        '--filename',
                        type=argparse.FileType('r'),
                        help='Image filename. Default is {}'.format(filename)
                       )
    parser.add_argument(
                        '--device',
                        type=str,
                        help='Device to be used (e.g., cuda, cpu, opencl). Default is {}'.format(device_name)
                       )
    parser.add_argument(
                        '--directory',
                        type=str,
                        help='Output directory location. Default is {}'.format(output_directory)
                       )
    parser.add_argument(
                        '--backlight',
                        type=str,
                        help='Backlight color primaries initialisation method. Default is {}'.format(backlight)
                       )

    parser.add_argument(
                        '--diopter',
                        type=float,
                        help='Unit used to measure the correction [- for myopia, + for hyperopia]. Default is {}'.format(diopter)
                       )
    
    args = parser.parse_args()
    if not isinstance(args.iterations, type(None)):
        no_of_iterations = args.iterations
    if not isinstance(args.filename, type(None)):
        filename = str(args.filename.name)
    if not isinstance(args.device, type(None)):
        device_name = str(args.device)
    if not isinstance(args.directory, type(None)):
        output_directory = str(args.directory)
    if not isinstance(args.backlight, type(None)):
        backlight = args.backlight
    if not isinstance(args.learningrate, type(None)):
        learning_rate = args.learningrate
    if not isinstance(args.diopter, type(None)):
        diopter = args.diopter
    device = torch.device(device_name)
    print('Filename: {}'.format(filename))
    print('Iterations: {}'.format(no_of_iterations))
    print('Learning rate: {}'.format(learning_rate))
    print('Device: {}'.format(device))
    print('Directory: {}'.format(output_directory))
    print('Backlight primary: {}'.format(backlight))
    print('prescription: {}'.format(prescription))
    print('diopter: {}'.format(diopter))
    process(
            no_of_iterations=no_of_iterations,
            filename=filename,
            directory=output_directory,
            backlight=backlight,
            learning_rate=learning_rate,
            kernel_type=kernel_type,
            device=device,
            diopter = diopter
           )


def process(
            no_of_iterations,
            filename,
            directory,
            backlight,
            device,
            diopter,
            learning_rate=0.01,
            kernel_fn='./output/kernel_lms.pt',
            kernel_type='LMS',
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
            ],             
            cylindrical_power=0,       
            axis_angle=0,           
            eye_pupil_diameter=6,
            psf_size_degrees=0.45  
           ):
    
    spherical_power = diopter 
    target_image = odak.learn.tools.load_image(filename)[:,:,0:3] / 255.
    if os.path.exists(kernel_fn):
        kernel = torch.load(kernel_fn)
    else:
        kernel = get_display_kernel_4D(
                                        target_image,
                                        kernel_type,
                                        zernike_coefficients=zernike_coefficients,
                                        spherical_power=spherical_power,
                                        cylindrical_power=cylindrical_power,
                                        axis_angle=axis_angle,
                                        eye_pupil_diameter=eye_pupil_diameter,
                                        psf_size_degrees=psf_size_degrees,
                                        backlight=backlight
                                        )
        odak.tools.check_directory('./output')
        torch.save(kernel, kernel_fn)
    optimizer = learned_prescription_optimizer(
                                               target_image,
                                               loss_function=None,
                                               no_of_iterations=no_of_iterations,
                                               learning_rate=learning_rate,
                                               directory=directory,
                                               kernel=kernel,
                                               kernel_type=kernel_type,
                                               device=device
                                             )
    optimized_image = optimizer.optimize(lms_space=True)


if "__main__":
    sys.exit(main())
