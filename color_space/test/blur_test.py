import torch
from odak.learn.perception import RadiallyVaryingBlur
from odak.learn.tools import load_image, save_image, resize
from odak.tools import check_directory
target_image = load_image( '/home/atlas/cll/learned_prescription_pre_zernike_correction/dataset/parrot_fig6_lms.png')
target_image = resize(target_image, 1)

output_directory = "./output/blur/"
check_directory(output_directory)


def blur_channels(input_image,channels = [1,1,1]):
    input_image_reshaped = (input_image.swapaxes(0,2)).unsqueeze(0)
    image_width = input_image.size(1) * 0.311 / 1000
    blur = RadiallyVaryingBlur()
    blurred_image = blur.blur(input_image_reshaped,alpha= 12.0, real_image_width= image_width, real_viewing_distance=2.4, centre = [0.5, 0.5], mode="quadratic")
    blurred_image = (blurred_image.swapaxes(1,3)).squeeze(0)
    save_image('{}/blurred.png'.format(output_directory), blurred_image)

    blurred_ch_1 = blurred_image[:,:,0] * channels[0] + input_image[:,:,0] * (1-channels[0])
    blurred_ch_2 = blurred_image[:,:,1] * channels[1] + input_image[:,:,1] * (1-channels[1])
    blurred_ch_3 = blurred_image[:,:,2] * channels[2] + input_image[:,:,2] * (1-channels[2])
    blurred_image = torch.cat((blurred_ch_1.unsqueeze(2),blurred_ch_2.unsqueeze(2),blurred_ch_3.unsqueeze(2)), dim=2)
    return blurred_image


blurred_image_re = blur_channels(input_image=target_image, channels = [1,1,1])
save_image('{}/blurred_recon.png'.format(output_directory), blurred_image_re)