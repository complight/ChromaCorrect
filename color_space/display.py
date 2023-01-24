import wave
import numpy as np_cpu
import torch
import odak
from torch.functional import F
from torch.nn import MSELoss
from color_space.retina import perceptual_color
from odak.learn.perception import RadiallyVaryingBlur


class sample_under_display():

    def __init__(self, resolution=[1920, 1080], distance_from_screen=800, pixel_pitch=0.311, read_spectrum='default', device=None):
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        self.read_spectrum = read_spectrum
        self.resolution = resolution
        self.distance_from_screen = distance_from_screen
        self.pixel_pitch = pixel_pitch
        self.retina = perceptual_color(self.device)
        self.l_normalised, self.m_normalised, self.s_normalised = self.retina.initialise_cones_normalised()
        self.lms_tensor = self.construct_matrix_lms(
                                                    self.l_normalised,
                                                    self.m_normalised,
                                                    self.s_normalised
                                                   )   
        
        self.color_1_mask, self.color_2_mask, self.luminance_mask = self.retina.third_channel_mask(self.resolution,
                                                                                                   self.distance_from_screen,
                                                                                                   self.pixel_pitch
                                                                                                   )
        self.dc = 5e-2
        self.mse_loss = MSELoss()
        self.radially_varying_blur = RadiallyVaryingBlur()
        return

    def __call__(self, input_image, ground_truth, gaze=None):
        """
        Evaluating an input image against a target ground truth image for a given gaze of a viewer.
        """
        # lms_image_second = self.rgb_to_lms(input_image.to(self.device))
        # lms_ground_truth_second = self.rgb_to_lms(ground_truth.to(self.device))
        lms_image_second = input_image.to(self.device)
        lms_ground_truth_second = ground_truth.to(self.device)
        lms_image_third = self.second_to_third_stage(lms_image_second)
        lms_ground_truth_third = self.second_to_third_stage(
            lms_ground_truth_second)
        loss_metamer_color = torch.mean(
            (lms_ground_truth_third - lms_image_third)**2)

        return loss_metamer_color
        # return self.mse_loss(input_image.double(), ground_truth.double())

    def initialise_rgb_backlight_spectrum(self):
        """
        Internal function to initialise baclight spectrum for color primaries. 
        """
        wavelength_range = np_cpu.linspace(400, 700, num=301)
        red_spectrum = [1 / (14.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (
            wavelength_range[i] - 650)**2 / (2 * 14.5**2)) for i in range(len(wavelength_range))]
        green_spectrum = [1 / (12 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (
            wavelength_range[i] - 550)**2 / (2 * 12.0**2)) for i in range(len(wavelength_range))]
        blue_spectrum = [1 / (12 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (
            wavelength_range[i] - 450)**2 / (2 * 12.0**2)) for i in range(len(wavelength_range))]

        red_spectrum = torch.from_numpy(
            red_spectrum / max(red_spectrum)) * 1.0
        green_spectrum = torch.from_numpy(
            green_spectrum / max(green_spectrum)) * 1.0
        blue_spectrum = torch.from_numpy(
            blue_spectrum / max(blue_spectrum)) * 1.0

        return red_spectrum.to(self.device), green_spectrum.to(self.device), blue_spectrum.to(self.device)

    def initialise_random_spectrum_normalised(self, dataset):
        """
        Initialise normalised light spectrum via combination of 3 gaussian distribution curve fitting [L-BFGS]. 
        Parameters
        ----------
        dataset                                : torch.tensor 
                                                 spectrum value against wavelength 
        peakspectrum                           :

        Returns
        -------
        light_spectrum                         : torch.tensor
                                                 Normalized light spectrum function       
        """
        if (type(dataset).__module__) == "torch":
            dataset = dataset.numpy()
        if dataset.shape[0] > dataset.shape[1]:
            dataset = np_cpu.swapaxes(dataset, 0, 1)
        x_spectrum = np_cpu.linspace(400, 700, num=301)
        y_spectrum = np_cpu.interp(x_spectrum, dataset[0], dataset[1])
        x_spectrum = torch.from_numpy(x_spectrum) - 550
        y_spectrum = torch.from_numpy(y_spectrum)
        max_spectrum = torch.max(y_spectrum)
        y_spectrum /= max_spectrum

        def gaussian(x, A=1, sigma=1, centre=0): return A * \
            torch.exp(-(x - centre)**2 / (2*sigma**2))

        def function(x, weights): return gaussian(
            x, *weights[:3]) + gaussian(x, *weights[3:6]) + gaussian(x, *weights[6:9])
        weights = torch.tensor(
            [1.0, 1.0, -0.2, 1.0, 1.0, 0.0, 1.0, 1.0, 0.2], requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weights], max_iter=1000, lr=0.1, line_search_fn=None)

        def closure():
            optimizer.zero_grad()
            output = function(x_spectrum, weights)
            loss = F.mse_loss(output, y_spectrum)
            loss.backward()
            return loss
        optimizer.step(closure)
        spectrum = function(x_spectrum, weights)
        return spectrum.detach().to(self.device)

    def initialise_normalised_spectrum_primaries(self,root = './backlight/'):
        '''
        Initialise normalised light spectrum via csv data and multilayer perceptron curve fitting. 

        Returns
        -------
        red_spectrum_fit                         : torch.tensor
                                                   Fitted red light spectrum function 
        green_spectrum_fit                       : torch.tensor
                                                   Fitted green light spectrum function  
        blue_spectrum_fit                        : torch.tensor
                                                   Fitted blue light spectrum function
        '''
        print(root)
        red_data = np_cpu.swapaxes(np_cpu.genfromtxt(
            root + 'red_spectrum.csv', delimiter=','), 0, 1)
        green_data = np_cpu.swapaxes(np_cpu.genfromtxt(
             root + 'green_spectrum.csv', delimiter=','), 0, 1)
        blue_data = np_cpu.swapaxes(np_cpu.genfromtxt(
             root + 'blue_spectrum.csv', delimiter=','), 0, 1)
        wavelength = np_cpu.linspace(400, 700, num=301)
        red_spectrum = torch.from_numpy(np_cpu.interp(
            wavelength, red_data[0], red_data[1])).unsqueeze(1)
        green_spectrum = torch.from_numpy(np_cpu.interp(
            wavelength, green_data[0], green_data[1])).unsqueeze(1)
        blue_spectrum = torch.from_numpy(np_cpu.interp(
            wavelength, blue_data[0], blue_data[1])).unsqueeze(1)
        wavelength = torch.from_numpy(wavelength).unsqueeze(1) / 550.
        curve = odak.learn.tools.multi_layer_perceptron(
            n_hidden=64).to(torch.device('cpu'))  # device
        curve.fit(wavelength.float(), red_spectrum.float(),
                  epochs=1000, learning_rate=5e-4)
        red_estimate = torch.zeros_like(red_spectrum)
        for i in range(red_estimate.shape[0]):
            red_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
        curve.fit(wavelength.float(), green_spectrum.float(),
                  epochs=1000, learning_rate=5e-4)
        green_estimate = torch.zeros_like(green_spectrum)
        for i in range(green_estimate.shape[0]):
            green_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
        curve.fit(wavelength.float(), blue_spectrum.float(),
                  epochs=1000, learning_rate=5e-4)
        blue_estimate = torch.zeros_like(blue_spectrum)
        for i in range(blue_estimate.shape[0]):
            blue_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
        primary_wavelength = torch.linspace(400, 700, 301)
        red_spectrum_fit = torch.cat(
            (primary_wavelength.unsqueeze(1), red_estimate), 1)
        green_spectrum_fit = torch.cat(
            (primary_wavelength.unsqueeze(1), green_estimate), 1)
        blue_spectrum_fit = torch.cat(
            (primary_wavelength.unsqueeze(1), blue_estimate), 1)
        red_spectrum_fit[:, 1] *= (red_spectrum_fit[:, 1] > 0)
        green_spectrum_fit[:, 1] *= (green_spectrum_fit[:, 1] > 0)
        blue_spectrum_fit[:, 1] *= (blue_spectrum_fit[:, 1] > 0)
        red_spectrum_fit=red_spectrum_fit.detach()
        green_spectrum_fit=green_spectrum_fit.detach()
        blue_spectrum_fit=blue_spectrum_fit.detach()
        return red_spectrum_fit[:, 1].to(self.device), green_spectrum_fit[:, 1].to(self.device), blue_spectrum_fit[:, 1].to(self.device)

    def display_spectrum_response(wavelength, function):
        """
        Internal function to provide light spectrum response at particular wavelength

        Parameters
        ----------
        wavelength                          : torch.tensor
                                              Wavelength in nm [400...700]
        function                            : torch.tensor
                                              Display light spectrum distribution function

        Returns
        -------
        ligth_response_dict                  : float
                                               Display light spectrum response value
        """
        wavelength = int(round(wavelength, 0))
        if wavelength >= 400 and wavelength <= 700:
            return function[wavelength - 400].item()
        elif wavelength < 400:
            return function[0].item()
        else:
            return function[300].item()

    def cone_response_to_spectrum(self, cone_spectrum, light_spectrum):
        """
        Internal function to calculate cone response at particular light spectrum. 

        Parameters
        ----------
        cone_spectrum                         : torch.tensor
                                                Spectrum, Wavelength [2,300] tensor 
        light_spectrum                        : torch.tensor
                                                Spectrum, Wavelength [2,300] tensor 


        Returns
        -------
        response_to_spectrum                  : float
                                                Response of cone to light spectrum [1x1] 
        """
        response_to_spectrum = torch.mul(cone_spectrum, light_spectrum)
        response_to_spectrum = torch.sum(response_to_spectrum)
        return response_to_spectrum.item()

    def construct_matrix_lms(self, l_response, m_response, s_response):
        '''
        Internal function to calculate cone  response at particular light spectrum. 

        Parameters
        ----------
        *_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)

        Returns
        -------
        lms_image_tensor                      : torch.tensor
                                                3x3 LMSrgb tensor

        '''
        if self.read_spectrum == 'default':
            red_spectrum, green_spectrum, blue_spectrum = self.initialise_rgb_backlight_spectrum()
            print('Estimated gaussian backlight is used')
        else:
            print('*.csv backlight data is used ')
            red_spectrum, green_spectrum, blue_spectrum = self.initialise_normalised_spectrum_primaries()
        l_r = self.cone_response_to_spectrum(l_response, red_spectrum)
        l_g = self.cone_response_to_spectrum(l_response, green_spectrum)
        l_b = self.cone_response_to_spectrum(l_response, blue_spectrum)
        m_r = self.cone_response_to_spectrum(m_response, red_spectrum)
        m_g = self.cone_response_to_spectrum(m_response, green_spectrum)
        m_b = self.cone_response_to_spectrum(m_response, blue_spectrum)
        s_r = self.cone_response_to_spectrum(s_response, red_spectrum)
        s_g = self.cone_response_to_spectrum(s_response, green_spectrum)
        s_b = self.cone_response_to_spectrum(s_response, blue_spectrum)
        self.lms_tensor = torch.tensor(
            [[l_r, m_r, s_r], [l_g, m_g, s_g], [l_b, m_b, s_b]]).to(self.device)
        return self.lms_tensor      



    def rgb_to_lms(self, rgb_image_tensor):
        """
        Internal function to calculate cone  response at particular light spectrum. 

        Parameters
        ----------
        rgb_image_tensor                      : torch.tensor
                                                Image RGB data to be transformed to LMS space


        Returns
        -------
        lms_image_tensor                      : float
                                              : Image LMS data transformed from RGB space [3xHxW]
        """
        image_flatten = torch.flatten(rgb_image_tensor, start_dim=0, end_dim=1)
        unflatten = torch.nn.Unflatten(
            0, (rgb_image_tensor.size(0), rgb_image_tensor.size(1)))
        converted_unflatten = torch.matmul(
            image_flatten.double(), self.lms_tensor.double())
        converted_image = unflatten(converted_unflatten)        
        # converted_image = self.hunt_adjustment(converted_image)
        return converted_image.to(self.device)

    def lms_to_rgb(self, lms_image_tensor):
        """
        Internal function to calculate cone  response at particular light spectrum. 

        Parameters
        ----------
        lms_image_tensor                      : torch.tensor
                                                Image LMS data to be transformed to RGB space


        Returns
        -------
       rgb_image_tensor                       : float
                                              : Image RGB data transformed from RGB space [3xHxW]
        """
        image_flatten = torch.flatten(lms_image_tensor, start_dim=0, end_dim=1)
        unflatten = torch.nn.Unflatten(
            0, (lms_image_tensor.size(0), lms_image_tensor.size(1)))
        converted_unflatten = torch.matmul(
            image_flatten.double(), self.lms_tensor.inverse().double())
        converted_rgb_image = unflatten(converted_unflatten)        
        return converted_rgb_image.to(self.device)


    def convert_to_lms(self, image_channel, wavelength, intensity):
        """
        Internal function to calculate cone  response at particular light spectrum intensity at particular wavelength. 

        Parameters
        ----------
        image_channel                         : torch.tensor
                                                Image color primary channel data to be transformed to LMS space.
        wavelength                            : float 
                                                Particular wavelength to be used in LMS conversion.
        intensity                             : float
                                                Particular intensity of color primary spectrum with respect to wavelength to be used in LMS conversion.


        Returns
        -------
        lms_image                             : torch.tensor 
                                                Image channel LMS data transformed from color primary to LMS space [HxWx3].
        """
        spectrum = torch.zeros(301)
        spectrum[wavelength-400] = intensity 
        l = self.cone_response_to_spectrum(self.l_normalised, spectrum)
        m = self.cone_response_to_spectrum(self.m_normalised, spectrum)
        s = self.cone_response_to_spectrum(self.s_normalised, spectrum)
        lms_tensor_wavelength = torch.tensor([l, m, s]).to(self.device)
        image_flatten = torch.flatten(image_channel, start_dim=0, end_dim=1)
        image_flatten = image_flatten.unsqueeze(0).swapaxes(0, 1)
        lms_tensor_wavelength = lms_tensor_wavelength.unsqueeze(0)
        lms_converted_flatten = torch.matmul(
            image_flatten.double(), lms_tensor_wavelength.double())
        unflatten = torch.nn.Unflatten(
            0, (image_channel.size(0), image_channel.size(1)))
        lms_image = unflatten(lms_converted_flatten)
        return lms_image.to(self.device)

 
    def second_to_third_stage(self, lms_image):
        '''
        This function turns second stage [L,M,S] values into third stage [(L+S)-M, M-(L+S), (M+S)-L]
        Equations are taken from Schmidt et al "Neurobiological hypothesis of color appearance and hue perception" 2014

        Parameters
        ----------
        lms_image                             : torch.tensor
                                                 Image data at LMS space (second stage)

        Returns
        -------
        third_stage                            : torch.tensor
                                                 Image data at LMS space (third stage)

        '''
        third_stage = torch.zeros(
            lms_image.shape[0], lms_image.shape[1], 3).to(self.device)
        third_stage[:, :, 0] = (lms_image[:, :, 1] +
                                lms_image[:, :, 2]) - lms_image[:, :, 0]
        third_stage[:, :, 1] = (lms_image[:, :, 0] +
                                lms_image[:, :, 2]) - lms_image[:, :, 1]
        third_stage[:, :, 2] = torch.sum(lms_image, dim=2) / 3.
        return third_stage


    def convert_to_second_stage(self,image_channel, wavelength, intensity):
       second_stage_image = self.second_to_third_stage(self.convert_to_lms(image_channel, wavelength, intensity))
       return second_stage_image.to(self.device)

    def gaussian_fovea_tensor(self, image_tensor, gaze_point=[0.5, 0.5], sigma=[180, 180], dc=1e-2):
        '''
        Internal function to create fovea tensor by using gaussian distribution. 

        Parameters
        ----------
        image_tensor                           : torch.tensor
                                                 Image data to be foveated
        gaze_point                             : list 
                                                 Input for gaze point's x and y. 
        sigma                                  : list 
                                                 Gaussian function standard deviation number to model fovea distribution
        dc                                     : float
                                                 Direct component to level zero values.


        Returns
        -------
        foveated_image                         : torch.tensor
                                                 Foveated image tensor. 

        '''
        gaze = [
            int((gaze_point[0] - 0.5) * image_tensor.shape[0]),
            int((gaze_point[1] - 0.5) * image_tensor.shape[1])
        ]
        fovea_gaze = odak.learn.tools.generate_2d_gaussian(
            kernel_length=[
                image_tensor.shape[0],
                image_tensor.shape[1]
            ],
            nsigma=sigma
        ).to(self.device)
        fovea_gaze = fovea_gaze
        fovea_gaze = fovea_gaze / fovea_gaze.max()
        fovea_gaze[fovea_gaze < dc] = dc
        fovea_gaze_padded = odak.learn.tools.zero_pad(fovea_gaze)
        fovea_gaze_padded_shifted = torch.roll(
            fovea_gaze_padded,
            shifts=(gaze[0], gaze[1]),
            dims=(0, 1)
        )
        fovea_gaze = odak.learn.tools.crop_center(fovea_gaze_padded_shifted)
        fovea_gaze = fovea_gaze[0:image_tensor.shape[0],
                                0:image_tensor.shape[1]]
        fovea_gaze = torch.stack([fovea_gaze, fovea_gaze, fovea_gaze], dim=2)
        return fovea_gaze.detach().clone()

    def blur_channels(self, input_image, channels=[1, 1, 1], gaze=None):
        '''
        Internal function to apply the radially varying blur to image channels.

        Parameters
        ----------
        image_tensor                           : torch.tensor
                                                 Image data to be blurred
        channels                               : list 
                                                 Input for channels to be blurred  
        gaze                                   : list 
                                                 Input for gaze point's x and y.

        Returns
        -------
        blurred_image                         : torch.tensor
                                                Blur applied image tensor. 
        '''
        input_image_reshaped = ((input_image.swapaxes(0, 2)).unsqueeze(0))
        image_width = input_image.size(1) * self.pixel_pitch / 1000
        blurred_image = self.radially_varying_blur.blur(input_image_reshaped, alpha=0.025,
                                                        real_image_width=image_width,
                                                        real_viewing_distance=self.distance_from_screen * 2 / 1000,
                                                        centre=gaze,
                                                        mode="quadratic"
                                                        )
        blurred_image = (blurred_image.swapaxes(1, 3)).squeeze(0)
        blurred_ch_1 = blurred_image[:, :, 0] * \
            channels[0] + input_image[:, :, 0] * (1-channels[0])
        blurred_ch_2 = blurred_image[:, :, 1] * \
            channels[1] + input_image[:, :, 1] * (1-channels[1])
        blurred_ch_3 = blurred_image[:, :, 2] * \
            channels[2] + input_image[:, :, 2] * (1-channels[2])
        blurred_image = torch.cat((blurred_ch_1.unsqueeze(2),
                                   blurred_ch_2.unsqueeze(2),
                                   blurred_ch_3.unsqueeze(2)), dim=2).to(self.device)
        return blurred_image

# -------------------------------------Hunt Adjustment / Tone Mapping -------------------------------------
    def hunt_adjustment(self,img):
        """
        Applies Hunt-adjustment to an image

        :param img: image tensor to adjust (with NxCxHxW layout in the L*a*b* color space)
        :return: Hunt-adjusted image tensor (with NxCxHxW layout in the Hunt-adjusted L*A*B* color space)
        """
        # Extract luminance component
        L = img[:, :, 0:1]
        # Apply Hunt adjustment
        img_h = torch.zeros(img.size()).to(self.device)
        img_h[:, :, 0:1] = L
        img_h[:, :, 1:2] = torch.mul((0.01 * L), img[:, :, 1:2])
        img_h[:, :, 2:3] = torch.mul((0.01 * L), img[:, :, 2:3])

        return img_h.to(self.device)

    def tone_map(self, img, tone_mapper, exposure):
        """
        Applies exposure compensation and tone mapping.
        Refer to the Visualizing Errors in Rendered High Dynamic Range Images
        paper for details about the formulas.

        :param img: float tensor (with NxCxHxW layout) containing nonnegative values
        :param tone_mapper: string describing the tone mapper to apply
        :param exposure: float tensor (with Nx1x1x1 layout) describing the exposure compensation factor
        """
        # Exposure compensation
        x = (2 ** exposure) * img.swapaxes(0,2).unsqueeze(0)

        # Set tone mapping coefficients depending on tone_mapper
        if tone_mapper == "reinhard":
            lum_coeff_r = 0.2126
            lum_coeff_g = 0.7152
            lum_coeff_b = 0.0722

            Y = x[:, 0:1, :, :] * lum_coeff_r + x[:, 1:2, :, :] * lum_coeff_g + x[:, 2:3, :, :] * lum_coeff_b
            return torch.clamp(torch.div(x, 1 + Y), 0.0, 1.0)

        if tone_mapper == "hable":
            # Source: https://64.github.io/tonemapping/
            A = 0.15
            B = 0.50
            C = 0.10
            D = 0.20
            E = 0.02
            F = 0.30
            k0 = A * F - A * E
            k1 = C * B * F - B * E
            k2 = 0
            k3 = A * F
            k4 = B * F
            k5 = D * F * F

            W = 11.2
            nom = k0 * torch.pow(W, torch.tensor([2.0]).cuda()) + k1 * W + k2
            denom = k3 * torch.pow(W, torch.tensor([2.0]).cuda()) + k4 * W + k5
            white_scale = torch.div(denom, nom)  # = 1 / (nom / denom)

            # Include white scale and exposure bias in rational polynomial coefficients
            k0 = 4 * k0 * white_scale
            k1 = 2 * k1 * white_scale
            k2 = k2 * white_scale
            k3 = 4 * k3
            k4 = 2 * k4
            # k5 = k5 # k5 is not changed
        else:
            # Source:  ACES approximation: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
            # Include pre-exposure cancelation in constants
            k0 = 0.6 * 0.6 * 2.51
            k1 = 0.6 * 0.03
            k2 = 0
            k3 = 0.6 * 0.6 * 2.43
            k4 = 0.6 * 0.59
            k5 = 0.14

        x2 = torch.pow(x, 2)
        nom = k0 * x2 + k1 * x + k2
        denom = k3 * x2 + k4 * x + k5
        denom = torch.where(torch.isinf(denom), torch.Tensor([1.0]).cuda(), denom)  # if denom is inf, then so is nom => nan. Pixel is very bright. It becomes inf here, but 1 after clamp below
        y = torch.div(nom, denom).squeeze(0).swapaxes(0,2)
        return torch.clamp(y, 0.0, 1.0)
#----------------------------------------CIE Lab Conversion-----------------------------------------------------
    def convert_to_Lab(self, lms_input_image): 
        lms_input_image = lms_input_image.swapaxes(0,2).unsqueeze(0)
        dim = lms_input_image.size()   
        inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.000000000]], [[0.918357670]]]).to(self.device)
        #---------------rgb2xyz---------------------
        a11 = 3.241003275
        a12 = -1.537398934
        a13 = -0.498615861
        a21 = -0.969224334
        a22 = 1.875930071
        a23 = 0.041554224
        a31 = 0.055639423
        a32 = -0.204011202
        a33 = 1.057148933

        A = torch.Tensor([[a11, a12, a13],
                          [a21, a22, a23],
                          [a31, a32, a33]])

        lms_input_image = lms_input_image.reshape(dim[0], dim[1], dim[2]*dim[3]).to(self.device)  # NC(HW)

        transformed_color = torch.matmul(A.double().to(self.device), lms_input_image.double())
        transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])
        #--------- xyz2lab-------------------------
        input_color = torch.mul(transformed_color, inv_reference_illuminant)
        delta = 6 / 29
        delta_square = delta * delta
        delta_cube = delta * delta_square
        factor = 1 / (3 * delta_square)

        clamped_term = torch.pow(torch.clamp(input_color, min=delta_cube), 1.0 / 3.0).to(dtype=input_color.dtype)
        div = (factor * input_color + (4 / 29)).to(dtype=input_color.dtype)
        input_color = torch.where(input_color > delta_cube, clamped_term, div)  # clamp to stabilize training

        L = 116 * input_color[:, 1:2, :, :] - 16
        a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])

        transformed_color = torch.cat((L, a, b), 1)
        transformed_color = transformed_color.squeeze(0).swapaxes(0,2)
        return transformed_color.to(self.device)
#----------------------------------------------------------------------------------------------
    def to(self, device):
        """
        Utilization function for setting the device.
        Parameters
        ----------
        device       : torch.device
                       Device to be used (e.g., CPU, Cuda, OpenCL).
        """
        self.device = device
        return self
