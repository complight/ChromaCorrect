import numpy as np_cpu
import torch
import math

class perceptual_color():

    def __init__(self, device=None):

        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        return

    def initialise_cones(self):
        """
        Initialise L,M,S cones as normal distribution with given sigma, and mu values. 

        Returns
        -------
        l_cone, m_cone, s_cone              : torch.tensor
                                              List of cone distribution.
        """
        wavelength_range = np_cpu.linspace(400, 700, num=301)
        dist_l = [1 / (32.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               567.5)**2 / (2 * 32.5**2)) for i in range(len(wavelength_range))]
        dist_m = [1 / (27.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               545.0)**2 / (2 * 27.5**2)) for i in range(len(wavelength_range))]
        dist_s = [1 / (17.0 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               447.5)**2 / (2 * 17.0**2)) for i in range(len(wavelength_range))]
        l_cone = torch.from_numpy(dist_l / max(dist_l))
        m_cone = torch.from_numpy(dist_m / max(dist_m))
        s_cone = torch.from_numpy(dist_s / max(dist_s))
        return l_cone, m_cone, s_cone

    def initialise_cones_normalised(self):
        """
        Internal function to nitialise normalised L,M,S cones as normal distribution with given sigma, and mu values. 

        Returns
        -------
        l_cone_n, m_cone_n, s_cone_n        : torch.tensor
                                              List of normalised cone distribution.
        """
        wavelength_range = np_cpu.linspace(400, 700, num=301)
        dist_l = [1 / (32.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               567.5)**2 / (2 * 32.5**2)) for i in range(len(wavelength_range))]
        dist_m = [1 / (27.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               545.0)**2 / (2 * 27.5**2)) for i in range(len(wavelength_range))]
        dist_s = [1 / (17.0 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               447.5)**2 / (2 * 17.0**2)) for i in range(len(wavelength_range))]

        l_cone_n = torch.from_numpy(dist_l/max(dist_l))
        m_cone_n = torch.from_numpy(dist_m/max(dist_m))
        s_cone_n = torch.from_numpy(dist_s/max(dist_s))
        return l_cone_n.to(self.device), m_cone_n.to(self.device), s_cone_n.to(self.device)

    def cone_response(wavelength, function):
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

    def third_channel_mask(self, resolution, distance_from_screen, pixel_pitch = 0.311):
        """
        Internal function to provide fovea masking by using measurements from reference:
        Hensen et al. `Color perception in the intermediate periphery of the visual field` 2009

        Parameters
        ----------
        image_tensor                        : torch.tensor
                                              Image tensor   
        distance_from_screen                : float
                                              Eye distance from the screen as mm
        gaze                                : list    
                                              Gaze location in x and y between [0,1]                                            

        Returns
        -------
        color_1_mask                        : torch.tensor
                                              Sensitivity tensor regarding to eccentiricity values (L-M in figure3 from reference)
        color_2_mask                        : torch.tensor
                                              Sensitivity tensor regarding to eccentiricity values (S-(L+M) in figure3 from reference)
        luminance_mask                      : torch.tensor
                                              Sensitivity tensor regarding to eccentiricity values (Lum in figure3 from reference)        
        """
        eccentiricity  = np_cpu.array([0., 10., 20., 30., 40., 50.])
        color_1_sensitivity = np_cpu.array([175./175., 133.1/175., 96.7/175., 86.5/175., 74.2/175., 55.8/175.])
        color_2_sensitivity = np_cpu.array([73.1/73.1, 45.1/73.1, 37.7/73.1, 49./73.1, 34.4/73.1, 8.4/73.1])
        luminance_sensitivity = np_cpu.array([94.3/94.3, 78.6/94.3, 66.6/94.3, 68.9/94.3, 61.5/94.3, 45.4/94.3])
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
        max_degree = math.floor(math.degrees(math.atan(screen_diagonal_distance / 2 / distance_from_screen)))  
        distance_range = torch.linspace(0,math.ceil(screen_diagonal_distance / 2), 100)     
        color_1_mask = torch.zeros_like(image_tensor)[:,:,0]
        color_2_mask = torch.zeros_like(image_tensor)[:,:,0]
        luminance_mask = torch.zeros_like(image_tensor)[:,:,0]
        for radius in distance_range:
                    radius_nonlinear = (radius * math.degrees(math.atan(radius / distance_from_screen)) / max_degree)
                    mask = (distance_from_center * pixel_pitch <= radius_nonlinear)
                    mask = mask.swapaxes(0,1)
                    current_angle = math.degrees(math.atan(radius_nonlinear / distance_from_screen))
                    color_1_mask[~mask] = np_cpu.interp(current_angle, eccentiricity, color_1_sensitivity)
                    color_2_mask[~mask] = np_cpu.interp(current_angle, eccentiricity, color_2_sensitivity)
                    luminance_mask[~mask] = np_cpu.interp(current_angle, eccentiricity, luminance_sensitivity)
        return color_1_mask.to(self.device), color_2_mask.to(self.device), luminance_mask.to(self.device)

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
