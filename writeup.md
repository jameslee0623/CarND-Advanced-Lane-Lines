## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

P.S. Some of the Parametrics and data structure are inspired by Eddie Forson's [article:](https://towardsdatascience.com/teaching-cars-to-see-advanced-lane-detection-using-computer-vision-87a01de0424f) 

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2-2]: ./output_images/undistorted_test1.jpg "Undistorted Road Transformed"
[image3]: ./output_images/color_binary.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/final_test1.jpg "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced Lane Finding.ipynb" (in function `load_obj_img_pts()`)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpts` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpts` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpts` and `imgpts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

After undistorted by function `undistort(img, mtx, dist)`, it brcame this:
![alt text][image2-2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination (`combined_binary(img)`) of color (`hls_binary(rgb_img, h_thr=(18,30), l_thr=(30,204), s_thr=(100,255), wl_thr=(210,255))`) and gradient (`gradient_binary(rgb_img, sx=(20, 120), sy=(50, 150), mag=(50, 150), ang=(.7, 1.3), kernel_size=15)`) thresholds to generate a binary image (thresholding steps at those functions).  Here's an example of my output for this step. Red pixels detected by `hls_binary(rgb_img)`; Green pixels detected by `gradient_binary(rgb_img)` function.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes 2 functions, one called `perspective_transform_matrices()`, which appears in the file `Advanced Lane Finding.ipynb`. Also, I call the `cv2.warpPerspective(img, M, img.shape[1::-1])` directly from OpenCV to transform images. The `perspective_transform_matrices()` function takes no inputs. The only purpose of it is calculate the matrices for perspective transform. I chose the hardcode the source `src_pts` and destination `dst_pts` points in the following manner:

```python
src_pts = np.array([
[left_btm, bottom_px],
[left_top, 450],
[right_top, 450], 
[right_btm, bottom_px]], np.float32)
#src_pts = np.array([[210,719], [596,450], [688,450], [1100, 719]], np.float32)

dst_pts = np.array([
[margin, bottom_px], 
[margin, 0], 
[right_px-margin, 0], 
[right_px-margin, bottom_px]], np.float32)
#dst_pts = np.array([[300,719], [300,0], [979,0], [979, 719]], np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 719      | 300, 719      | 
| 596, 450      | 300, 0        |
| 688, 450      | 979, 0        |
| 1100, 719     | 979, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I use `find_lane_pixels(binary_warped)` to find lane pixels. Then call the `np.polyfit(lefty, leftx, 2)` to fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this by function `compute_cur(left_line, right_line)` in `Advanced Lane Finding.ipynb`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_lane_area(binary_warped, left_line, right_line, undist_img)`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I didn't implement the "memory system" in this project yet. So it has to use `histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)` to find the lane lines frame by frame. It will waste some computing power on searching the lane line evry frame.
The other problem is the "outliers". My pipeline falsely detected the shadow of the short wall on the left hand side in few frames (around 3 frames out of 50 mins videos). Hopefully it could be solved by the "memory system" as well.