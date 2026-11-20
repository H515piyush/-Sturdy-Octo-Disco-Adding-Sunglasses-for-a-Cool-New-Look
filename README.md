# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

### NAME : Piyush Kumar

### REG NO : 212223220075

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

## Program:
```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread('pass.JPeG')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

faceImage.shape

#resized_faceImage.shape
faceImage.shape

# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('glass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = cv2.cvtColor(glassPNG, cv2.COLOR_BGR2GRAY)

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');

# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[190:240,150:340] = glassBGR
plt.imshow(faceWithGlassesNaive[...,::-1])

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[135:185,110:300]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")

# Make a fresh copy of the original image
faceWithGlassesArithmetic = faceImage.copy()

# Create a mask to remove the white background
glassGray = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)
_, glassMask = cv2.threshold(glassGray, 240, 255, cv2.THRESH_BINARY_INV)

# Get the inverse mask
glassMaskInv = cv2.bitwise_not(glassMask)

# Select the eye region
y1, y2 = 200, 250
x1, x2 = 200, 390

roi = faceWithGlassesArithmetic[y1:y2, x1:x2]

# Apply sunglasses without the white background
roi_bg = cv2.bitwise_and(roi, roi, mask=glassMaskInv)
glass_fg = cv2.bitwise_and(glassBGR, glassBGR, mask=glassMask)

# Combine both images
faceWithGlassesArithmetic[y1:y2, x1:x2] = cv2.add(roi_bg, glass_fg)

# Display the final result
plt.figure(figsize=[20,20])

plt.subplot(121)
plt.imshow(faceImage[:,:,::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(faceWithGlassesArithmetic[:,:,::-1])
plt.title("With Sunglasses")

plt.show()

```
## Output
<img width="628" height="557" alt="Screenshot 2026-11-20 104434" src="https://github.com/user-attachments/assets/7b43a5db-8e90-4cff-a90c-55ec2109106e" />
<img width="832" height="462" alt="Screenshot 2026-11-20 104440" src="https://github.com/user-attachments/assets/956fa9f6-175c-4eaf-9334-0ad485ba4047" />
<img width="1423" height="250" alt="Screenshot 2026-11-20 104448" src="https://github.com/user-attachments/assets/28e77f70-7a9d-474a-bfc4-03bce38b247a" />
<img width="643" height="551" alt="Screenshot 2026-11-20 104455" src="https://github.com/user-attachments/assets/f23a46d6-c282-4eb8-b310-c7a3d252d2d5" />
<img width="1457" height="202" alt="Screenshot 2026-11-20 104503" src="https://github.com/user-attachments/assets/cfd59586-7822-4ffa-aceb-c6d929d0cb41" />
<img width="1417" height="812" alt="Screenshot 2026-11-20 104518" src="https://github.com/user-attachments/assets/6e916cf7-7703-4595-bf6f-96c296c99844" />



