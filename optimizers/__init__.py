from asyncore import read
import torch
import odak
from tqdm import tqdm
from torchvision import transforms
import kornia
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from color_space.display import sample_under_display



class learned_prescription_optimizer():

    """
    A configurable class for optimizing learned prescriptions
    """

    def __init__(self, target_image, no_of_iterations=1000,
                 learning_rate=0.01, loss_function=None,
                 optimization_mode='Stochastic Gradient Descent',
                 max_val=1.,
                 point_spread_function=None,
                 directory='./output',
                 kernel=None,
                 kernel_type=None,
                 device=None
                ):
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        torch.cuda.empty_cache()
        torch.random.seed()
        odak.tools.check_directory(directory)
        self.target_image = target_image.detach().to(self.device)
        self.input_image = self.target_image.detach().clone().to(self.device).requires_grad_()
        self.no_of_iterations = no_of_iterations
        self.learning_rate = learning_rate
        self.optimization_mode = optimization_mode
        self.directory = directory
        self.relu = torch.nn.ReLU()
        self.max_val = max_val
        self.kernel_type = kernel_type
        self.init_loss_function(loss_function)
        self.init_optimizer()
        if kernel_type == 'camera':
            self.init_prescription(self.camera_kernel_4D(kernel))
            print('camera kernel is active')
        else:
            self.init_prescription(kernel)
        self.l1_loss = torch.nn.L1Loss()
        

    def init_prescription(self, kernel, debug=False):
        """
        Internal function to initialize presecription.
        """
        self.kernel = kernel
        if isinstance(self.kernel, type(None)):
            self.kernel = self.sample_kernel()
        if len(self.kernel.shape) < 4:
            self.kernel= self.kernel.unsqueeze(0)
        kernel_magnitude = abs(self.kernel)
        for i in range(kernel_magnitude.shape[0]):
            odak.learn.tools.save_image(
                                        '{}/psf_{:04d}.png'.format(self.directory, i),
                                        kernel_magnitude[i],
                                        cmin=0.,
                                        cmax=float(kernel_magnitude.max())
                                       )


    def camera_kernel_4D(self,rgb_kernel):
        print(rgb_kernel.shape)
        rgb_kernel = rgb_kernel.to(self.device)

        rgb_kernel_red = torch.zeros_like(rgb_kernel).to(self.device)
        rgb_kernel_red[:,:,0] = rgb_kernel[:,:,0]
        rgb_camera_kernel_red = self.camera_converstion(rgb_kernel_red)

        rgb_kernel_green = torch.zeros_like(rgb_kernel).to(self.device)
        rgb_kernel_green[:,:,1] = rgb_kernel[:,:,1]
        rgb_camera_kernel_green = self.camera_converstion(rgb_kernel_green)

        rgb_kernel_blue = torch.zeros_like(rgb_kernel).to(self.device)
        rgb_kernel_blue[:,:,2] = rgb_kernel[:,:,2]
        rgb_camera_kernel_blue = self.camera_converstion(rgb_kernel_blue)

        rgb_camera_kernel_4D = torch.cat((rgb_camera_kernel_red.unsqueeze(0),
                                        rgb_camera_kernel_green.unsqueeze(0),
                                        rgb_camera_kernel_blue.unsqueeze(0)),
                                        dim=0
                                        )                          
        return rgb_camera_kernel_4D


        
    def camera_converstion(self, kernel):
        c_rgb_tensor = torch.tensor([[0.9917, 0.1517, 0.1310],
                                    [0.0020, 0.9796, 0.367],
                                    [0.0000, 0.1436, 0.9282]]).to(self.device)
        image_flatten = torch.flatten(kernel, start_dim=0, end_dim=1)
        unflatten = torch.nn.Unflatten(
            0, (kernel.size(0), kernel.size(1)))
        converted_unflatten = torch.matmul(
            image_flatten.double(), c_rgb_tensor.double())
        converted_kernel = unflatten(converted_unflatten)        
        return converted_kernel.to(self.device)


    def sample_kernel(self, sigma=[0.3, 0.3, 0.3]):
        """
        Internal function to create a gaussian kernel of MxNx3 resolution.
        """
        kernel = torch.stack(
                             [
                              odak.learn.tools.generate_2d_gaussian(kernel_length=[self.input_image.shape[0], self.input_image.shape[1]], nsigma=[sigma[0], sigma[0]]),
                              odak.learn.tools.generate_2d_gaussian(kernel_length=[self.input_image.shape[0], self.input_image.shape[1]], nsigma=[sigma[1], sigma[1]]),
                              odak.learn.tools.generate_2d_gaussian(kernel_length=[self.input_image.shape[0], self.input_image.shape[1]], nsigma=[sigma[2], sigma[2]])
                             ],
                             dim=2
                            ).to(self.device)
        kernel[kernel < 0] = 0.
        kernel[kernel > 0] = kernel.max()
        return kernel


    def init_optimizer(self):
        """
        Internal function to set the optimizer.
        """
        if self.optimization_mode == 'Stochastic Gradient Descent':
            self.optimizer = torch.optim.Adam(
                                              [self.input_image],
                                              lr=self.learning_rate,
                                             )


    def init_loss_function(self, loss_function=None):
        """
        Internal function to set the loss function
        """
        self.loss_function = loss_function
        self.loss_type = 'custom'
        if isinstance(self.loss_function, type(None)):
            self.loss_function = torch.nn.MSELoss()
            self.loss_type = 'naive'


    def evaluate(self, simulated_image, target_image):
        """
        Internal function to evaluate the loss

        """
        image_input = (simulated_image - simulated_image.min()) / (simulated_image.max() - simulated_image.min())
        image_target = (target_image - target_image.min()) / (target_image.max() - target_image.min())
        loss = self.loss_function(image_input, image_target)
        return loss 


    def optimize(self,lms_space=False):
        """
        Function to optimize learned prescriptions

        Parameters
        ----------
        kernel        : torch.tensor
                        Kernel to optimize against.
        """
        if self.optimization_mode == 'Stochastic Gradient Descent':
            optimized_input_image = self.stochastic_gradient_descent()
        return optimized_input_image


    def spatial_gradient_loss(self, input_image):
        spatial_gradient_input = kornia.filters.spatial_gradient(input_image.swapaxes(0,2).unsqueeze(0))
        spatial_grad_x_input = spatial_gradient_input.squeeze(0).swapaxes(0,1)[0,:,:,:]
        spatial_grad_x_input = spatial_grad_x_input.swapaxes(0,2)
        spatial_grad_y_input = spatial_gradient_input.squeeze(0).swapaxes(0,1)[1,:,:,:]
        spatial_grad_y_input = spatial_grad_y_input.swapaxes(0,2)                  
        gradient_loss = torch.mean(torch.abs(spatial_grad_x_input)+ torch.abs(spatial_grad_y_input))
        return gradient_loss

    def lpip_loss(self,target_image ,input_image):
        image_input = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        image_target = (target_image - target_image.min()) / (target_image.max() - target_image.min())
        input_image =image_input.swapaxes(0,2).unsqueeze(0)
        target_image = image_target.swapaxes(0,2).unsqueeze(0)     
        lpip_loss = self.lpips(target_image,input_image)
        return lpip_loss


    def forward(self, input_image, kernel_type):
        """
        Function to convolve a gaussian kernel point spread function (PSF) over an image.

        Parameters
        ----------
        input_image                : torch.tensor
                                     Input image (MxNx3 resolution)

        Returns
        -------
        convoled_lms                : torch.tensor
                                     Image blurred by convolution of PSF kernel on LMS space
        """
        result = torch.zeros_like(self.kernel).to(self.device)
        convolved_lms = torch.zeros_like(input_image).to(self.device)    
        for i in range(self.kernel.shape[0]):                     
            kernel_response_color_primary = self.kernel[i,:,:,:].to(self.device)                 
            image = odak.learn.tools.zero_pad(input_image[:, :, i])
            for j in range(3):
                kernel = odak.learn.tools.zero_pad(kernel_response_color_primary[:, :, j])
                fft_input_image = torch.fft.fftshift(torch.fft.fft2(image))
                fft_kernel = torch.fft.fftshift(torch.fft.fft2(kernel))
                U = (fft_input_image * fft_kernel).to(self.device) 
                result[i, :, :, j] += odak.learn.tools.crop_center(torch.abs(torch.fft.ifftshift(torch.fft.ifft2(U))))
        convolved_lms[:,:,0] = result[0,:,:,0] + result[1,:,:,0] + result[2,:,:,0]
        convolved_lms[:,:,1] = result[0,:,:,1] + result[1,:,:,1] + result[2,:,:,1]
        convolved_lms[:,:,2] = result[0,:,:,2] + result[1,:,:,2] + result[2,:,:,2]
        return convolved_lms 
    

    def stochastic_gradient_descent(self):
        """
        Function to optimize learned prescriptions using stochastic gradient descent

        Returns
        -------
        input_image       : torch.tensor
                            optimized input image
        """
        if self.kernel.shape[0] == 1:
            target = self.target_image.float()
        else:            
            self.color_spectrum = sample_under_display(device=self.device,read_spectrum='default')
            if self.kernel_type == 'camera':
                target = self.camera_converstion(self.target_image).float()
            else:
                target =self.color_spectrum.rgb_to_lms(self.target_image).float()
        t = tqdm(range(self.no_of_iterations), leave=False)
        for step in t:            
            self.optimizer.zero_grad()            
            input_image = torch.clamp(self.input_image,0,1)
            simulated_result = self.forward(input_image,kernel_type=self.kernel_type) 
            loss = self.evaluate(simulated_result, target)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            description = "Stochastic Gradient Descent, loss:{:.4f}".format(loss.item())
            t.set_description(description)
            if step % 10 == 0 or step == self.no_of_iterations - 1:
                odak.learn.tools.save_image('{}/cache_input_{:04d}.png'.format(self.directory, step),input_image, cmin=0., cmax=1.)        
                simulated_normalized = (simulated_result-simulated_result.min()) / (simulated_result.max()- simulated_result.min())
                odak.learn.tools.save_image('{}/cache_simulated_{:04d}.png'.format(self.directory, step),simulated_normalized, cmin=0., cmax=1.)
        print(description)
        blurred_target = self.forward(target.detach().clone(),kernel_type=self.kernel_type)
        blurred_target = (blurred_target - blurred_target.min()) / (blurred_target.max() - blurred_target.min())
        odak.learn.tools.save_image('{}/cache_blurred_original.png'.format(self.directory), blurred_target, cmin=0., cmax=float(blurred_target.max()))
        target = (target - target.min()) / (target.max() - target.min())
        odak.learn.tools.save_image('{}/cache_target.png'.format(self.directory), target, cmin=0., cmax=float(self.target_image.max()))
        
        return input_image
