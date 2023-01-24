
import torch 
import odak
import matplotlib.pyplot as plt
from odak.learn.tools import load_image, save_image, resize
from odak.tools import check_directory
import numpy as np
import math


output_directory = "./output/fovea"
check_directory(output_directory)

target_image = load_image( './data/car.png')
target_image = resize(target_image, 1)


def third_channel_mask(resolution = [1920,1080], distance_from_screen = 500, pixel_pitch = 0.311, gaze=None):
    eccentiricity  = np.array([0., 10., 20., 30., 40., 50.])
    color_1_sensitivity = np.array([175./175., 133.1/175., 96.7/175., 86.5/175., 74.2/175., 55.8/175.])
    color_2_sensitivity = np.array([73.1/73.1, 45.1/73.1, 37.7/73.1, 49./73.1, 34.4/73.1, 8.4/73.1])
    luminance_sensitivity = np.array([94.3/94.3, 78.6/94.3, 66.6/94.3, 68.9/94.3, 61.5/94.3, 45.4/94.3])
    gaze = [
            int((gaze[0]-0.5) * resolution[1]),
            int((gaze[1]-0.5) * resolution[0])
           ]
    image_tensor = torch.zeros(resolution[1]*2,resolution[0]*2,1)
    distance_x = 0.5 * image_tensor.size(1)
    distance_y = 0.5 * image_tensor.size(0)
    grid_x = torch.arange(image_tensor.size(1))
    grid_y = torch.arange(image_tensor.size(0))
    X, Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    distance_from_center = torch.sqrt((X - distance_x)**2 + (Y -distance_y)**2)
    image_height_screen = image_tensor.size(0) * pixel_pitch
    image_width_screen = image_tensor.size(1) * pixel_pitch
    screen_diagonal_distance =  math.sqrt(image_height_screen**2 + image_width_screen**2)
    max_degree = math.floor(math.degrees(math.atan(screen_diagonal_distance/2/distance_from_screen)))  
    distance_range = torch.linspace(0,math.ceil(screen_diagonal_distance/2),30)     
    color_1_mask = torch.zeros_like(image_tensor)[:,:,0]
    color_2_mask = torch.zeros_like(image_tensor)[:,:,0]
    luminance_mask = torch.zeros_like(image_tensor)[:,:,0]
    for radius in distance_range:
                radius_nonlinear = (radius * math.degrees(math.atan(radius/distance_from_screen))/max_degree)
                mask = (distance_from_center * pixel_pitch   <= radius_nonlinear)
                mask = mask.swapaxes(0,1)
                current_angle = math.degrees(math.atan(radius_nonlinear/distance_from_screen))
                color_1_mask[~mask] = current_angle
                color_2_mask[~mask] = np.interp(current_angle,eccentiricity,color_2_sensitivity)
                luminance_mask[~mask] = np.interp(current_angle,eccentiricity,luminance_sensitivity)
    color_1_mask_cropped = odak.learn.tools.crop_center(torch.roll(color_1_mask, shifts=(gaze[0], gaze[1]), dims=(0, 1)))
    color_2_mask_cropped = odak.learn.tools.crop_center(torch.roll(color_2_mask, shifts=(gaze[0], gaze[1]), dims=(0, 1)))
    luminance_mask_cropped = odak.learn.tools.crop_center(torch.roll(luminance_mask, shifts=(gaze[0], gaze[1]), dims=(0, 1)))
    return color_1_mask, color_1_mask_cropped, color_2_mask_cropped, luminance_mask_cropped

color_1_mask , color_1_mask_cropped, color_2_mask_cropped, luminance_mask_cropped = third_channel_mask([1920,1080],800,0.311,[0.,0.])
save_image('{}/color_1_mask.png'.format(output_directory), color_1_mask * 3)
save_image('{}/color_1_mask_cropped.png'.format(output_directory), color_1_mask_cropped*3)
save_image('{}/color_2_mask_cropped.png'.format(output_directory), color_2_mask_cropped*128)
save_image('{}/luminance_mask_cropped.png'.format(output_directory), luminance_mask_cropped*128)       

print(torch.max(color_1_mask))
print(torch.max(color_1_mask_cropped))