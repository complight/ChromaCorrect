from display import sample_under_display
import torch
import sys
from tqdm import tqdm
from odak.learn.tools import load_image, save_image, resize
from odak.tools import check_directory

relu = torch.nn.ReLU()

def optimise(target_image, number_of_iterations=200):
    '''
    Image reconstruction via MSELoss minimization
    '''
    loss_fn = torch.nn.MSELoss()
    x = torch.rand_like(target_image.double()) * target_image.max()
    x = x.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([x], lr=0.075)
    t = tqdm(range(number_of_iterations), leave=False)
    for i in t:
        optimizer.zero_grad()
        loss = color_display(x, target_image.double(), gaze = [0.5,0.5])
        loss.backward()
        optimizer.step()
        description = 'loss:{:.4f}'.format(loss.item())
        description = "loss:{:.4f}, min:{:.4f}, max:{:.4f}".format(loss.item(), x.min(), x.max())
        t.set_description(description)
        if loss.item() == 0:
            break
        if i % 20 == 0 or i == number_of_iterations - 1:
            save_image('{}/difference_{:04}.png'.format(output_directory, i), (torch.abs(target_image - x) * 50) * 255.)
            save_image('{}/solution_{:04}.png'.format(output_directory, i), x * 255.)     
    print(description)
    return x

sys.path.append('../')


target_image = load_image( './data/car.png')[:,:,0:3] / 255.
target_image = resize(target_image, 1)

output_directory = "./output/"
check_directory(output_directory)

device_ = torch.device("cpu")
color_display = sample_under_display(read_spectrum='default', device=device_)


niter = 100
constructed_image = optimise(target_image, niter)






