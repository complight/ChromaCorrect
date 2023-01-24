## color_space

Creating a class for color perception research,  © __Computational Light Laboratory at University College London__

### loss.py Module


There is a **loss.py**. Under this script, we have a class named **perceptual_color**.

L,M,S cone distributions against wavelength are modelled by using normal distribution. 

Approach : **perceptual_color** has 3 methods to create, manipulate and plot data for L,M,S as lists. 
  
                > **initialise_cones_normalised()** to initialise normalised L,M,S distributions as torch tensors.    
                > **cone_response()** to get normalised responsitivity spectra (NRS) via given wavelength value             
                > **plot_response()** to plot L,M,S wavelength vs NRS

Fitting is below. Reference data :  "Stockman, MacLeod & Johnson 2-deg cone fundamentals".


Testing 1st Approach : ```python3 test_lms_1.py```


![LMS_correlation](https://user-images.githubusercontent.com/46696280/176939803-8a1d5d57-2e0b-4cfb-851a-2bd41b85819f.png)


### display.py Module

There is a **display.py**. Under this script, we have a class named **sample_under_display**. 

Display light spectrum against wavelength are modelled by combined normal (gaussian) distribution with L-BFGS algorithm (torch optimizer)

Approach : **sample_under_display** has 3 methods to create, manipulate and plot data for random *.csv read color primary channel. 

                > **initialise_random_spectrum_normalised()** to initialise normalised display spectrum distributions as torch tensors.   
                > **display_spectrum_response()** to get display spectrum via given wavelength value between 400 and 700            
                > **plot_response()** to plot display spectrum wavelength vs NRS
                
Dashed line is target.
 
Blue line is fitted tensor. 

![Figure_1](https://user-images.githubusercontent.com/46696280/177022864-60b0080d-abd9-408b-bd17-f935564d6f31.png)

#### **display.cone_response_to.spectrum()** function

This function is implemented to get L,M,S float values after -> torch.mul(cone_response,light_spectrum) -> torch.sum(resultant)

![image](https://user-images.githubusercontent.com/46696280/177063734-5427f888-1fb5-4708-be4a-e6bbfa64d166.png)


                > L cone response to display spectrum - Red : tensor(44.4775, dtype=torch.float64)
                > M cone response to display spectrum - Red : tensor(44.1414, dtype=torch.float64)
                > S cone response to display spectrum - Red : tensor(29.6639, dtype=torch.float64)


#### **initialise_rgb_backlight_spectrum()** function

This function is under display.py, and it creates the LED backlight spectrum via gaussian distribution.

![RGB_correlation](https://user-images.githubusercontent.com/46696280/177237669-a9952134-e0ac-480a-bee5-c1f66d427d61.png)

#### **def construct_matrix_lms()** function 

This function creates 3x3 LMSrgb matrix 

![image](https://user-images.githubusercontent.com/46696280/177237958-e0f00101-2b9d-43fd-906a-99797e61c9f2.png)


#### **rgb_2_lms_linear()** function 

This function is under display.py, and it converts rgb space to lms space. 

![image](https://user-images.githubusercontent.com/46696280/177238462-db0a5edc-a6bf-45cc-ab51-4e121fd29aca.png)

#### __call__ () function

Call function to calculate MSE loss betweeb two given image. 

    

    l_normalised, m_normalised, s_normalised = perceptual_color.initialise_cones_normalised()
    color_display = sample_under_display(l_normalised,m_normalised,s_normalised)    
    loss = color_display(rgb_image,rgb_image) 
    
    >> tensor(0., dtype=torch.float64)




