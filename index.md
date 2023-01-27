

<h1 align="center">ChromaCorrect: Prescription Correction in Virtual Reality Headsets through Perceptual Guidance</h1>

 <p align="center">
  <img width ="10%" src="https://user-images.githubusercontent.com/46696280/214999103-a8a33456-b66c-4cb4-8102-578379a7e2c9.png">
  <img width ="10%" src="https://user-images.githubusercontent.com/46696280/214999398-0ed2f644-4983-4841-af23-b1d7221fc34d.png">
  <img width ="10%" src="https://user-images.githubusercontent.com/46696280/214999444-b96281cd-9158-416f-a6da-f82169780886.png">
  <img width ="10%" src="https://user-images.githubusercontent.com/46696280/214999478-ea45353d-3704-4290-8e90-10c747887253.png">
 </p>



 <h3>
  <p align="center">
    <a href="https://aguzel.github.io/">Ahmet H. Güzel*<sup>1</sup></a> |
    <a href="https://www.linkedin.com/in/jeanne-beyazian/?trk=public_profile_browsemap&originalSubdomain=uk">Jeanne Beyazian<sup>2</sup></a> |
    <a href="https://www.cs.unc.edu/~cpk/">Praneeth Chakravarthula<sup>3</sup></a> |
    <a href="https://kaanaksit.com/">Kaan Akşit<sup>2</sup></a>
   </p>
 </h3>
 
   <p align="center">
    <strong><sup>1</sup>University of Leeds,  <sup>3</sup>Princeton University,  <sup>2</sup>University College London </strong>
   </p>


   <p align="center">
    <a href="https://complightlab.com/">UCL Computational Light Laboratory*<sup>1</sup></a> |
   </p>

  
<h2>
 <p align="center">
   <a href="https://arxiv.org/abs/2212.04264/">Paper</a> |
   <a href="https://github.com/complight/ChromaCorrect">Code</a> |
   <a href="https://www.youtube.com/watch?v=fjexa7ga-tQ">Video</a>
 </p>
</h2>
  
 <p align="center">
  <img src="https://user-images.githubusercontent.com/46696280/214193337-b6f80d66-bfa4-4025-b63e-0400a0b50969.png">
 </p> 
 
 <h2>
 <p align="center">
    Abstract
 </p>
 </h2>
   
A large portion of today’s world population suffer from vision impairments and wear prescription eyeglasses. However, eyeglasses causes additional bulk and discomfort when used with augmented and virtual reality headsets, thereby negatively impacting the viewer’s visual experience. In this work, we remedy the usage of prescription
eyeglasses in Virtual Reality (VR) headsets by shifting the optical complexity completely into software and propose a prescriptionaware rendering approach for providing sharper and immersive VR imagery. To this end, we develop a differentiable display and visual perception model encapsulating display-specific parameters, color and visual acuity of human visual system and the user-specific refractive errors. Using this differentiable visual perception model,
we optimize the rendered imagery in the display using stochastic gradient-descent solvers. This way, we provide prescription glassesfree sharper images for a person with vision impairments. We evaluate our approach on various displays, including desktops and VR headsets, and show significant quality and contrast improvements
for users with vision impairments. 

 <h2>
 <p align="center">
    Optimization Pipeline
 </p>
 </h2>
<p align="center" width="100%">
    <img width="70%" src="https://user-images.githubusercontent.com/46696280/215001021-3bfde19c-eadf-49df-96ce-1b7ec51e0445.png">
</p>


<h3> Step 1: </h3>  A screen with color primaries (RGB) displays an input image.

<h3> Step 2: </h3> A viewer’s eye images the displayed image onto the retina with a unique Point Spread Function (PSF) describing the optical aberrations of that person’s eye.

<h3> Step 3: </h3> Retinal cells convert the aberrated RGB image to a trichromat sensation, also known as Long-Medium-Short (LMS) cone perception

<h3> Step 4: </h3> Our optimization pipeline relies on the perceptually guided model described in previous steps (1-3). Thus, the optimization pipeline converts a given RGB image to LMS space at each optimization step while accounting for the PSFs of a viewer modelled using Zernike polynomials.

<h3> Step 5: </h3> Our loss function penalizes the simulated image derived from the perceptually guided model against a target image in LMS space. Finally, our differentiable optimization pipeline identifies proper input RGB images using a Stochastic Gradient Descent solver.

 <h2>
 <p align="center">
    Evaluation
 </p>
 </h2>  

### 1) Hardware Setup

#### 1.a) Conventional Display 

For every experimental image capture, we fixed the pose, ISO, and focus setting of the camera to ensure a consistent view with a nearsighted prescription of -1.50.

<p align="center" width="100%">
<img width="50%" src="https://user-images.githubusercontent.com/46696280/214996949-e35725f4-5ad2-4237-ba68-e5f14f0d4797.png">
</p>

#### 1.b) Virtual Reality Headset

(A) We use a virtual reality headset and a camera to capture images from our virtual reality headset. To emulate a prescription problem in the visual system, we use a defocus lens. (B) We take pictures with fixed pose and camera focus from behind the defocus lens to evaluate reconstructed images.
 
<p align="center" width="100%">
<img width="50%" src="https://user-images.githubusercontent.com/46696280/214997797-adb2a5c7-7449-4ec1-b0b1-f9fef8581439.png">
</p>

#### 1.b) Results

<p align="center" width="100%">
<img width="60%" src="https://user-images.githubusercontent.com/46696280/214997968-09149daf-fea5-48b2-8546-737242fbea33.png">
</p>

ChromaCorrect improves the conventional approach by means of color and contrast. 


### 2) Simulated 

In the second part, we evaluated our method with different prescriptions to model various refractive eye problems. Thus, all the images used in this part are evaluated in simulated LMS space.

#### 2.a) Results

<p align="center" width="100%">
<img width="70%" src="https://user-images.githubusercontent.com/46696280/215000048-ede10d3a-f1e8-4add-bebc-1d3bda72f2a7.png">
</p>
Here we compare outputs from five different refractive vision problems (myopia, hyperopia, hyperopic astigmatism, myopic astigmatism, and myopia with
hyperopic astigmatism) for five sample input images. We provide simulated LMS space representations of target image, conventional method output, and our method. FLIP
per-pixel difference along with it’s mean value (lower is better), SSIM and PSNR are provided to compare performance of methods. Our method shows better loss numbers
for each image quailty metrics for each experiment in simulated LMS space. The contrast improvement by using our method against conventional method also can be
obvserved perceptually.

 <h2>
 <p align="center">
    Learned Model (Neural ChromaCorrect)
 </p>
 </h2>  
 
We implement a semi-supervised deep learning model capable of reconstructing optimized  images from their original RGB versions. We use a U-Net architecture [40] for this purpose. Such a solution is more suitable than an iterative process for achieving real-time applications.

<p align="center" width="100%">
<img width="40%" src="https://user-images.githubusercontent.com/46696280/215000422-fb8e8cf4-ad3e-42fe-b163-53f2f1cd9b69.png">
</p>
The learned model significantly reduces image generation time, with an average of 2.9 milliseconds per corrected image compared to the original method’s 8.127
seconds, a speed increase of approximately 2800 times

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
