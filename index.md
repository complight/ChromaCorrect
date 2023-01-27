

<h2 align="center">ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance</h2>
 
 <p align="center">
  <b>Authors</b><br>
  <a href="https://aguzel.github.io/">Ahmet H. Güzel*<sup>1</sup></a> |
  <a href="https://www.linkedin.com/in/jeanne-beyazian/?trk=public_profile_browsemap&originalSubdomain=uk">Jeanne Beyazian<sup>2</sup></a> |
  <a href="https://www.cs.unc.edu/~cpk/">Praneeth Chakravarthula<sup>3</sup></a> |
  <a href="https://kaanaksit.com/">Kaan Akşit<sup>2</sup></a>
  <br><br>
  <sup>1</sup>University of Leeds, <sup>3</sup>Princeton University, <sup>2</sup>University College London
 </p>
 
<h2> 
 <p align="center">
  <a href="https://arxiv.org/abs/2212.04264/">Paper</a> |
  <a href="https://github.com/complight/ChromaCorrect">Code</a> |
  <a href="https://www.youtube.com/watch?v=fjexa7ga-tQ">Video<a>
 </p>
</h2>
  

<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/46696280/214193337-b6f80d66-bfa4-4025-b63e-0400a0b50969.png">
</p>

## Abstract

A large portion of today’s world population suffer from vision impairments and wear prescription eyeglasses. However, eyeglasses causes additional bulk and discomfort when used with augmented and virtual reality headsets, thereby negatively impacting the viewer’s visual experience. In this work, we remedy the usage of prescription
eyeglasses in Virtual Reality (VR) headsets by shifting the optical complexity completely into software and propose a prescriptionaware rendering approach for providing sharper and immersive VR imagery. To this end, we develop a differentiable display and visual perception model encapsulating display-specific parameters, color and visual acuity of human visual system and the user-specific refractive errors. Using this differentiable visual perception model,
we optimize the rendered imagery in the display using stochastic gradient-descent solvers. This way, we provide prescription glassesfree sharper images for a person with vision impairments. We evaluate our approach on various displays, including desktops and VR headsets, and show significant quality and contrast improvements
for users with vision impairments. 


## Our Optimization Pipeline 
<p align="center" width="100%">
    <img width="70%" src="https://user-images.githubusercontent.com/46696280/214984308-f67c3a9b-11f0-4d81-8f3f-4319fca1b266.png">
</p>


 **1** ) A screen with color primaries (RGB) displays an input image.

 **2** ) A viewer’s eye images the displayed image onto the retina with a unique Point Spread Function (PSF) describing the optical aberrations of that person’s eye.

 **3** ) Retinal cells convert the aberrated RGB image to a trichromat sensation, also known as Long-Medium-Short (LMS) cone perception

 **4** ) Our optimization pipeline relies on the perceptually guided model described in previous steps (1-3). Thus, the optimization pipeline converts a given RGB image to LMS space at each optimization step while accounting for the PSFs of a viewer modelled using Zernike polynomials.

 **5** ) Our loss function penalizes the simulated image derived from the perceptually guided model against a target image in LMS space. Finally, our differentiable optimization pipeline identifies proper input RGB images using a Stochastic Gradient Descent solver.




## Citation

```
@article{https://doi.org/10.48550/arxiv.2212.04264,
  doi = {10.48550/ARXIV.2212.04264},  
  url = {https://arxiv.org/abs/2212.04264},  
  author = {Güzel, Ahmet and Beyazian, Jeanne and Chakravarthula, Praneeth and Akşit, Kaan},  
  title = {ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}

```
