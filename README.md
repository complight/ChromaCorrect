# ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance
### [Project Page](https://github.com/complight/ChromaCorrect) | [Video]() | [Paper](https://arxiv.org/abs/2212.04264)

[ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance](https://arxiv.org/abs/2212.04264)  
 [Ahmet H. Güzel](https://aguzel.github.io/)\*<sup>1</sup>,
 [Jeanne Beyazian](https://www.linkedin.com/in/jeanne-beyazian/?trk=public_profile_browsemap&originalSubdomain=uk/)<sup>2</sup>,
 [Praneeth Chakravarthula](https://www.cs.unc.edu/~cpk/)<sup>3</sup>,
 [Kaan Akşit](https://kaanaksit.com/)<sup>2</sup>,
 
 <sup>1</sup>University of Leeds, <sup>3</sup>Princeton University, <sup>2</sup>University College London
 
 ![image](https://user-images.githubusercontent.com/46696280/214193337-b6f80d66-bfa4-4025-b63e-0400a0b50969.png)

 
## What is ChromaCorrect?

A large portion of today’s world population suffer from vision impairments and wear prescription eyeglasses. However, eyeglasses causes additional bulk and discomfort when used with augmented and virtual reality headsets, thereby negatively impacting the viewer’s visual experience. In this work, we remedy the usage of prescription
eyeglasses in Virtual Reality (VR) headsets by shifting the optical complexity completely into software and propose a prescriptionaware rendering approach for providing sharper and immersive VR imagery. To this end, we develop a differentiable display and visual perception model encapsulating display-specific parameters, color and visual acuity of human visual system and the user-specific refractive errors. Using this differentiable visual perception model,
we optimize the rendered imagery in the display using stochastic gradient-descent solvers. This way, we provide prescription glassesfree sharper images for a person with vision impairments. We evaluate our approach on various displays, including desktops and VR headsets, and show significant quality and contrast improvements
for users with vision impairments.

## Description

This repository contains our implementation for learning prescriptions related to refractive vision problems (myopia, presbyopia, astigmatism etc.).

Clone the entire repository and navigate to the root directory.

```shell
git clone --recurse-submodules git@github.com:complight/learned_prescription.git
cd learned_prescription
```

## Install the required dependencies

`requirements.txt` can help you to install the required packages using `pip`:

```shell
pip3 install -r requirements.txt
```

## Running the code (performing optimization)

Once you have the requirements successfully installed, you are ready to run the optimisation.

```shell
python3 main.py --device cuda --filename dataset/parrot.png --directory sample --backlight read
```

You can also adjust the parameters used in the optimization routine by passing arguments. To learn more about parameters:

```shell
python3 main.py --help
```
## Dataset 

You can use images from both **Dataset** and **Dataset_D2K** for your experiments. 

Dataset : Image datasets released by Computational Light Lab. at UCL. 
Dataset_D2K : Image dataset shared by DIV2K image dataset. 

## Importing Display Backlight Data

File names have to be named exactly as below : 
- red_spectrum.csv
- green_spectrum.csv
- blue_spectrum.csv

inside .csv, format should be below (column1 : wavelength, column2 : spectrum data)

    column1     column2
    400         spectrum[0]
    .
    .
    .
    700         spectrum[i]


## Citation

```
@article{https://doi.org/10.48550/arxiv.2212.04264,
  doi = {10.48550/ARXIV.2212.04264},  
  url = {https://arxiv.org/abs/2212.04264},  
  author = {Güzel, Ahmet and Beyazian, Jeanne and Chakravarthula, Praneeth and Akşit, Kaan},  
  keywords = {Human-Computer Interaction (cs.HC), Graphics (cs.GR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, I.3.3; I.2.10},  
  title = {ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}

```
