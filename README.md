# MultiStyle Sematic Style Transfer
This project is aimed to transfer different sematic objects in one image into different styles.

- [x] On-line Image Optimization
- [x] Add Total Variation Loss (encourages spatial smoothness in the generated image)
- [ ] MRF-based blending
- [ ] Off-line Model optimization
- [ ] Video style tranfer

## Procedure
![image](https://github.com/aa10402tw/MultiStyle_Sematic_Style_Transfer/blob/master/images/procedure.png) <br>

## Result
### Compare with naive approach
![image](https://github.com/aa10402tw/MultiStyle_Sematic_Style_Transfer/blob/master/results/people2_bright_dark/Vanilla_Sematic.jpg) 
<br>
### Style Blending Loss
![image](https://github.com/aa10402tw/MultiStyle_Sematic_Style_Transfer/blob/master/results/people2_bright_dark/Blend_Ratio_0.60.jpg) <br>

<!-- ### 1D visualization
![image](https://github.com/aa10402tw/GAN_visualization/blob/master/result/1D.gif =250x250) <br>
In 1D visualization, the red/blue line are representing the Probability Density Function for data generating from real/generator. <br>
And the dot line are the output for discriminator, where the higher value mean the discriminator believes the data is from real distribution more. <br>


### 2D visualization
![image](https://github.com/aa10402tw/GAN_visualization/blob/master/result/2D.gif =250x250) <br>
In 2D visualization, the red/blue dots are the data points generating from real/generator. <br>
And the contour line are the output for discriminator, where the higher value mean the discriminator believes the data is from real distribution more. <br> -->

