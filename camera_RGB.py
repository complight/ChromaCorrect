import torch
import numpy
from torchvision import transforms
from PIL import Image
from odak.learn.tools import load_image, save_image
from odak.tools import check_directory
import sys
from tqdm import tqdm
sys.path.append('../')
sys.path.append('./')
from color_space.display import sample_under_display




output_directory = "../output/camere_rgb_space/"
check_directory(output_directory)

rescale = transforms.Compose([transforms.Resize((512,512))])
red = Image.open('/home/bellerophon/Documents/UCL-ComputationalLightLab/learned_prescription/color_space/data/red2.jpg')
green = Image.open('/home/bellerophon/Documents/UCL-ComputationalLightLab/learned_prescription/color_space/data/green2.jpg')
blue = Image.open('/home/bellerophon/Documents/UCL-ComputationalLightLab/learned_prescription/color_space/data/blue2.jpg')
img = Image.open('/home/bellerophon/Documents/UCL-ComputationalLightLab/learned_prescription/color_space/data/parrot.png')

# red = rescale(red)
# green = rescale(green)
# blue = rescale(blue)
img = rescale(img)

img = torch.from_numpy(numpy.array(img)) / 255.0
red_cam = torch.from_numpy(numpy.array(red)) / 255.0
green_cam = torch.from_numpy(numpy.array(green)) / 255.0
blue_cam = torch.from_numpy(numpy.array(blue)) / 255.0
red_display = torch.zeros_like(red_cam) 
red_display[:,:,0] = 1
green_display = torch.zeros_like(green_cam)
green_display[:,:,1] = 1
blue_display = torch.zeros_like(blue_cam)
blue_display[:,:,2] = 1



# rgb_cam = red_cam + green_cam + blue_cam
# rgb_display = red_display + green_display + blue_display

# save_image('{}/rgb_cam.png'.format(output_directory), rgb_cam)
# save_image('{}/rgb_display.png'.format(output_directory), rgb_display)


def mult_func(conv_tensor, rgb_image_tensor):
    image_flatten = torch.flatten(rgb_image_tensor, start_dim=0, end_dim=1)
    unflatten = torch.nn.Unflatten(
        0, (rgb_image_tensor.size(0), rgb_image_tensor.size(1)))
    converted_unflatten = torch.matmul(
        image_flatten.double(), conv_tensor.double())
    converted_image = unflatten(converted_unflatten)
    return converted_image


# loss = torch.nn.MSELoss()
# conv_matrix = torch.zeros((3,3),requires_grad=True)
# optimizer = torch.optim.Adam([conv_matrix], lr=0.01)
# t = tqdm(range(300), leave=False)
# for i in t:
#     optimizer.zero_grad()
#     input_image = mult_func(conv_tensor=conv_matrix,rgb_image_tensor=blue_display)
#     criterion = loss(input_image.double(), blue_cam.double())
#     criterion.backward()
#     optimizer.step()
#     description = 'loss:{:.4f}'.format(criterion.item())
#     description = "loss:{:.4f}, min:{:.4f}, max:{:.4f}".format(criterion.item(), conv_matrix.min(), conv_matrix.max())
#     t.set_description(description)  
# print(description)
# print(conv_matrix)
# final_matrix = torch.tensor([[0.9917, 0.1517, 0.1310],
#                              [0.0020, 0.9796, 0.367],
#                              [0.0000, 0.1436, 0.9282]])


# red_response = torch.matmul(torch.tensor([1,0,0]).float(), final_matrix)
# green_response = torch.matmul(torch.tensor([0,1,0]).float(), final_matrix)
# blue_response = torch.matmul(torch.tensor([0,0,1]).float(), final_matrix)
# rgb_response = red_response + green_response + blue_response
# print(rgb_response)
# print(final_matrix)


output_directory = "./output/camere_rgb_space/"
check_directory(output_directory)
save_image('{}/parrot2.png'.format(output_directory),img * 255.0)
color_spectrum = sample_under_display()
red_cam_res = color_spectrum.rgb_to_Crgb(red_display)
red_cam_res /= red_cam_res.max()
save_image('{}/rgb_red_converted.png'.format(output_directory), red_cam_res * 255.0)
save_image('{}/rgb_red_target.png'.format(output_directory),red_cam * 255.0)

green_cam_res = color_spectrum.rgb_to_Crgb(green_display)
green_cam_res /= green_cam_res.max()
save_image('{}/rgb_green_converted.png'.format(output_directory), green_cam_res * 255.0)
save_image('{}/rgb_green_target.png'.format(output_directory),green_cam * 255.0)

blue_cam_res = color_spectrum.rgb_to_Crgb(blue_display)
blue_cam_res /= blue_cam_res.max()

save_image('{}/rgb_blue_converted.png'.format(output_directory), blue_cam_res * 255.0)
save_image('{}/rgb_blue_target.png'.format(output_directory),blue_cam * 255.0)