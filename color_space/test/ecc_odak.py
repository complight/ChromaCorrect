import torch
from odak.learn.perception.metamer_mse_loss import MetamerMSELoss
from odak.learn.perception import foveation
from odak.learn.tools import load_image, save_image, resize
from odak.tools import check_directory
input_image = load_image( './data/car.png')
image_width = 1920 * 0.311 / 1000
output_directory = "./output/foveation_odak/"
check_directory(output_directory)

ecc_map, distance_map= foveation.make_eccentricity_distance_maps(gaze_location = (0.5,0.5),
                                                    image_pixel_size=(1080,1920),
                                                    real_image_width=image_width,
                                                    real_viewing_distance=0.8
                                                    )

save_image('{}/ecc_map.png'.format(output_directory), ecc_map * 20 )