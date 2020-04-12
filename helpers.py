import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=14):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # Find center of image
    sizeY, sizeX = img.shape
    center_X = sizeX/2;
    center_Y = sizeY/2;
    # devide lines based on the horizontal center of the image
    left_lines = lines[lines[:,0,0] <= center_X]
    right_lines = lines[lines[:,0,0] > center_X]
    # extrapolating lines
    extrap_left=extrapolate_lines(left_lines,sizeY,center_Y)
    extrap_right=extrapolate_lines(right_lines,sizeY,center_Y)
    # Creating empty image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Sanity check, if equals -1 it means that no lines were detected
    if (type(extrap_left)==int) | (type(extrap_right)==int):
        return line_img
    
    draw_lines(line_img, extrap_left)
    draw_lines(line_img, extrap_right)
    return line_img

def extrapolate_lines(lines, bottom_frame, center_frame):
    """
        'lines' is a vector of the form: [x1,y1,x2,y2] which contains 
        the two points to create a line. Shape of the vector = [:,0,4]
        
        'bottom frame' is the location of the bottom of the image
        
        'center_frame' is the location of the vertical center of the image
        
        Returns resultVect a single line coordinates from the bottom of the image 
        in the form: [[[x1,y1,x2,y2]]]
    """
    # we will calculate the slope "m" and the intercept "b" of each one of the lines
    # avarage them and return a single "m" and "b" value
    # eq for "m": m = x1-x2/y1-y2
    # eq for "b": b=Y-mX where Y=y2 or y1 and X=x2 or x1
    avg_m=0;
    avg_b=0;
    numberOfLines = len(lines)
    if numberOfLines == 0:
        return -1
    for line in lines:
        for x1,y1,x2,y2 in line:
            m=(y1-y2)/(x1-x2)
            if m==0:
                return -1
            avg_m=avg_m+m
            avg_b=avg_b+y1-m*x1
    avg_m=avg_m/numberOfLines;
    avg_b=avg_b/numberOfLines;
    # Now we get the coordinates from the bottom based on the avaraged parameters
    # Y=avg_m*X+avg_b where X=[bottom_frame-avg_b]/avg_m
    # we also get the coordinate from almost the center of the image in the same way
    # X=[center_frame*1.2-avg_b]/avg_m
    Y1=int(bottom_frame);
    X1=int((Y1-avg_b)/avg_m);
    Y2=int(center_frame*1.2);
    X2=int((Y2-avg_b)/avg_m);
    resultVect=[[[X1,Y1,X2,Y2]]]
    return resultVect
    
# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)



### 
## ******************************************* Finding lines functions *******************************************


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    height, width = np.shape(img)
    bottom_half = img[np.int(height/2):,:]

    # Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0) ## *Note 
    return histogram, bottom_half


def starting_pos_lines(img):
    # Take a histogram of the bottom half of the image
    histogram, _ = hist(img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    hist_Lenght = histogram.shape[0]
    midpoint = np.int(hist_Lenght//2)#two slashes to apply floor to the division. i.e. floor(3/2) = 3//2
    # Find position of left peak and right peak
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base

   

def window_boundaries(binary_warped_img, curr_window, window_height, margin, left_center, right_center):
    """
        This function creates the windows for the left and right lines
        
        Output: dictionary with the boundaries of the windows
    """
    warped_img_height = binary_warped_img.shape[0]

    win_y_low = warped_img_height - (curr_window+1)*window_height
    win_y_high = win_y_low + window_height
    win_xLeft_low = left_center-margin
    win_xLeft_high = left_center+margin
    win_xRight_low = right_center-margin
    win_xRight_high = right_center+margin
    dictWindow = {'y_high': win_y_high, 'y_low': win_y_low, 'xLeft_low': win_xLeft_low,
                 'xLeft_high': win_xLeft_high, 'xRight_low': win_xRight_low, 'xRight_high': win_xRight_high}
    return dictWindow

def draw_windows(out_img, xLeft_low, xLeft_high, xRight_low, xRight_high, y_low, y_high):
    """
        This function draws rectangles on the out_img based on the opposite corners coordinates of
        the rectangle
    """
    cv2.rectangle(out_img, (xLeft_low,y_low), (xLeft_high, y_high), color = (0,255,0), thickness = 2)
    cv2.rectangle(out_img, (xRight_low,y_low), (xRight_high, y_high), color = (0,255,0), thickness = 2)
    
    return out_img

def active_pixels_window(img_act_pixels_x, img_act_pixels_y, xLeft_low, 
                         xLeft_high, xRight_low, xRight_high, y_low, y_high):
    """
        This function outputs a list of active pixels in the current window
    """
    # nonzero function returns the indexes of all true values as false is equivalent to zero
    win_act_pixels_left__ind = np.nonzero((img_act_pixels_x >= xLeft_low) & (img_act_pixels_x < xLeft_high) & (
                            img_act_pixels_y >= y_low) & (img_act_pixels_y < y_high))[0] 

    win_act_pixels_right__ind = np.nonzero((img_act_pixels_x >= xRight_low) & (img_act_pixels_x < xRight_high) & (
                            img_act_pixels_y >= y_low) & (img_act_pixels_y < y_high))[0] 
    
    return win_act_pixels_left__ind, win_act_pixels_right__ind

def update_window_centers(left_center, right_center, win_act_pixels_left__ind, win_act_pixels_right__ind, 
                          img_act_pixels_x, minpix):
    """
        Update current window center based on minpix parameter
        
        If found > minpix pixels, recenter next window on their mean position
    """

    if len(win_act_pixels_left__ind) > minpix:
        new_left_center = np.int(np.mean(img_act_pixels_x[win_act_pixels_left__ind]))
    else:
        new_left_center = left_center
        
    if len(win_act_pixels_right__ind) > minpix:
        new_right_center = np.int(np.mean(img_act_pixels_x[win_act_pixels_right__ind]))
    else:
        new_right_center = right_center
        
    return new_left_center, new_right_center

def sliding_windows(binary_warped_img, nwindows, margin, minpix, visualize=False):
    """
        This function creates a list with the position (x,y) of the pixels that are part of the lines
    """
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped_img.shape[0]//nwindows)
    
    # Get the indices of all nonzero elements in the image. AKA activated pixels
    nonzeroY, nonzeroX = binary_warped_img.nonzero();
    nonzeroX = np.array(nonzeroX) # activated pixels indexes in the X direction
    nonzeroY = np.array(nonzeroY) # activated pixels indexes in the Y direction
    
    # setting starting position lines as current position for each window, they could be updated later
    leftx_base, rightx_base = starting_pos_lines(binary_warped_img)
    leftx_current_center = leftx_base
    rightx_current_center = rightx_base
    
    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped_img, binary_warped_img, binary_warped_img))*255).astype(np.uint8)
    
    # List of all active pixels index in all windows for left and right lines
    left_act_pixels__ind = []
    right_act_pixels__ind = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        
        # get curret window boundaries
        windowData = window_boundaries(binary_warped_img, window, window_height, margin, 
                          left_center=leftx_current_center, right_center=rightx_current_center)
        
        # Visualize windows in case visualize is set to True
        if visualize:
            out_img = draw_windows(out_img, windowData['xLeft_low'], windowData['xLeft_high'], 
                                   windowData['xRight_low'], windowData['xRight_high'], 
                                   windowData['y_low'], windowData['y_high'])

        # Identify all active pixels in the window
        win_act_pixels_left__ind, win_act_pixels_right__ind = active_pixels_window(
                             nonzeroX,  nonzeroY,
                            windowData['xLeft_low'], windowData['xLeft_high'], 
                            windowData['xRight_low'], windowData['xRight_high'], 
                            windowData['y_low'], windowData['y_high'])
        
        # Populate lists of all active pixels in all windows from left and right lines
        left_act_pixels__ind = np.concatenate((left_act_pixels__ind, win_act_pixels_left__ind))
        right_act_pixels__ind = np.concatenate((right_act_pixels__ind, win_act_pixels_right__ind))
        # parse to int indexes must always be integers
        left_act_pixels__ind = left_act_pixels__ind.astype(np.int)
        right_act_pixels__ind = right_act_pixels__ind.astype(np.int)
        
        # Update current window center based on minpix parameter
        leftx_current_center, rightx_current_center= update_window_centers(leftx_current_center, rightx_current_center, 
                                                                    win_act_pixels_left__ind, win_act_pixels_right__ind,
                                                                           nonzeroX, minpix)
   
    # return left and right line pixel positions
    if visualize:
        return (nonzeroX[left_act_pixels__ind], nonzeroY[left_act_pixels__ind], 
            nonzeroX[right_act_pixels__ind], nonzeroY[right_act_pixels__ind], out_img)
    else:
        return (nonzeroX[left_act_pixels__ind], nonzeroY[left_act_pixels__ind], 
            nonzeroX[right_act_pixels__ind], nonzeroY[right_act_pixels__ind])

def fit_polynomial(binary_warped, visualize=False, meters=False, m_per_pix_X=0, m_per_pix_Y=0):
    if visualize:
        left_posX, left_posY, right_posX, right_posY, out_img = sliding_windows(binary_warped, 9, 100, 50, visualize=visualize)
        out_img[left_posY, left_posX] = [255, 0 , 0] # Setting all pixels in left windows as red
        out_img[right_posY, right_posX] = [0, 0 , 255] # Setting all pixels in left windows as blue
    else:
        left_posX, left_posY, right_posX, right_posY = sliding_windows(binary_warped, 9, 100, 50)
    
    # Fit a 2nd order polynomial
    left_fit_coeff = np.polyfit(left_posY, left_posX, 2)
    right_fit_coeff = np.polyfit(right_posY, right_posX, 2)
    
    y = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitX = left_fit_coeff[0]*y**2+left_fit_coeff[1]*y+left_fit_coeff[2]
    right_fitX = right_fit_coeff[0]*y**2+right_fit_coeff[1]*y+right_fit_coeff[2]
    
    if visualize:
        # plotting fitted polynomial to the image.
        plt.figure(figsize=(700/1200*25,1200/700*25))
        plt.title('Sliding window + fitted polynomial visualization', fontsize=40)
        plt.plot(left_fitX, y, color='yellow')
        plt.plot(right_fitX, y, color='yellow')
        plt.imshow(out_img)
    if meters:
        # To return real-world data 
        left_fit_coeff = np.polyfit(left_posY*m_per_pix_Y, left_posX*m_per_pix_X, 2)
        right_fit_coeff = np.polyfit(right_posY*m_per_pix_Y, right_posX*m_per_pix_X, 2)
        return left_fit_coeff, right_fit_coeff
    else:
        # To return pixels data
        return left_fit_coeff, right_fit_coeff

def poly_boundaries_pixels(warped_image, prev_left_poly_coeffs, prev_right_poly_coeffs, margin):
    """
        This function extracts the active pixels around the polynomials from the left and right lines
        
        Output: dictionary with the boundaries of the active pixels around polynomials
    """
    # Get the indices of all nonzero elements in the image. AKA activated pixels
    activePixels_posY, activePixels_posX = warped_image.nonzero()
    activePixels_posX = np.array(activePixels_posX) # activated pixels indexes in the X direction
    activePixels_posY = np.array(activePixels_posY) # activated pixels indexes in the Y direction
    
    # polynomial left line coefficients
    L2 = prev_left_poly_coeffs[0]
    L1 = prev_left_poly_coeffs[1]
    L0 = prev_left_poly_coeffs[2]
    
    # polynomial right line coefficients
    R2 = prev_right_poly_coeffs[0]
    R1 = prev_right_poly_coeffs[1]
    R0 = prev_right_poly_coeffs[2]

    # Get indexes for the polynomial boundaries LEFT
    poly_bound_active_left__ind = (activePixels_posX > (L2*activePixels_posY**2+L1*activePixels_posY+L0-margin)) & (
    activePixels_posX < (L2*activePixels_posY**2+L1*activePixels_posY+L0+margin))
    # Get indexes for the polynomial boundaries RIGHT
    poly_bound_active_right__ind = (activePixels_posX > (R2*activePixels_posY**2+R1*activePixels_posY+R0-margin)) & (
    activePixels_posX < (R2*activePixels_posY**2+R1*activePixels_posY+R0+margin))
    
    # extract positions from indexes and active pixels
    leftLine_act_posX = activePixels_posX[poly_bound_active_left__ind]
    leftLine_act_posY = activePixels_posY[poly_bound_active_left__ind]
    rightLine_act_posX = activePixels_posX[poly_bound_active_right__ind]
    rightLine_act_posY = activePixels_posY[poly_bound_active_right__ind]

    # storing info into a dictionary
    Act_poly_boundaries_dict = {'leftLine_act_posX': leftLine_act_posX, 'leftLine_act_posY': leftLine_act_posY,
                               'rightLine_act_posX': rightLine_act_posX, 'rightLine_act_posY': rightLine_act_posY}
    
    return Act_poly_boundaries_dict



def search_around_poly(warped_image, prev_left_poly_coeffs, prev_right_poly_coeffs, margin_around_poly, visualize=False):
    # getting all pixels inside the polynomial boundaries
    polyBounded_dict = poly_boundaries_pixels(warped_image, prev_left_poly_coeffs, 
                                              prev_right_poly_coeffs, margin_around_poly)
    
    # get the new polynomial for current frame that will be used in next frame as well
    # Fit a 2nd order polynomial
    curr_left_poly_coeffs = np.polyfit(polyBounded_dict['leftLine_act_posY'], 
                                       polyBounded_dict['leftLine_act_posX'], 2)
    curr_right_poly_coeffs = np.polyfit(polyBounded_dict['rightLine_act_posY'], 
                                        polyBounded_dict['rightLine_act_posX'], 2)
    
    ## Visualization
    # In case we want to visualize the outcome we do all the process to plot it
    if visualize:
        # Creating y-axis
        y = np.array(list(range(warped_image.shape[0])))
        # Calc x-axis using the new coefficients for left and right lines, respectevely
        left_fitX = curr_left_poly_coeffs[0]*y**2 + curr_left_poly_coeffs[1]*y + curr_left_poly_coeffs[2]
        right_fitX = curr_right_poly_coeffs[0]*y**2 + curr_right_poly_coeffs[1]*y + curr_right_poly_coeffs[2]
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((warped_image, warped_image, warped_image))*255
        ## Color the pixels that are inside the window of prior polynomial curve boundaries
        # Left line pixels are colored in red
        out_img[polyBounded_dict['leftLine_act_posY'], polyBounded_dict['leftLine_act_posX']] = [255,0,0]
        # Right line pixels are colored in blue
        out_img[polyBounded_dict['rightLine_act_posY'], polyBounded_dict['rightLine_act_posX']] = [0,0,255]
        ## Color green the section that is part of the window generated by previous polynomial
        ## To do so we need to use cv2.fillPoly which draws an area based on the perimeter you provide.
        ## Please refer to figure 1 to understand the idea.
        # 1st we will get another image like out_img so that we can plot the green shape in it.
        window_img = np.zeros_like(out_img)
        ## Now we need to put all data in the required format for fillPoly as explained in figure 1 above
        # The following line sets the coordinates for the left Border of the green shape in figure 1.
        left_line_leftBorder = np.array([np.transpose(np.vstack([left_fitX-margin_around_poly, y]))])
        # The following line sets the coordinates for the right Border of the green shape in figure 1. Please note
        #   that here we are adding the np.flipud which order the coordinates Upside-Down to follow the purple arrows 
        #   in figure 1
        left_line_rightBorder = np.array([np.flipud(np.transpose(np.vstack([left_fitX+margin_around_poly, y])))])
        # now we concatenate the cordinates of both borders to get a single array of coordinates
        left_line_pts = np.hstack((left_line_leftBorder, left_line_rightBorder))
        
        ## We do the same for the right line
        right_line_leftBorder = np.array([np.transpose(np.vstack([right_fitX-margin_around_poly, y]))])
        right_line_rightBorder = np.array([np.flipud(np.transpose(np.vstack([right_fitX+margin_around_poly, y])))])
        right_line_pts = np.hstack((right_line_leftBorder, right_line_rightBorder))
        
        # We now draw the lane onto the empty "window_img" image, we set the color of the shape as green (0,255,0)
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        
        # We can now plot both images using a cross-dissolved technique which morphs images by weighting one
        #  over the other. In this case we will merge the out_img and the window_img
        mergedPlot = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Finally we plot the new polynomial lines onto the lane lines as well as the mergedPlot
        plt.figure(figsize=(700/1200*25,1200/700*25))
        plt.plot(left_fitX, y, color='yellow')
        plt.plot(right_fitX, y, color='yellow')
        plt.imshow(mergedPlot.astype(np.uint8))
        plt.title('Polynomial search technique', fontsize=35)
        ## End of visualization steps ##
    
        
    return curr_left_poly_coeffs, curr_right_poly_coeffs

def measure_curvature(y_eval, left_poly_coeffs, right_poly_coeffs):
    '''
    @Description: Calculates the curvature of polynomial functions in pixels.

    @param y_eval: the "y" value to which we want to evaluate
    @param left_poly_coeffs, right_poly_coeffs: coefficients of the polynomial fit for left and right lines.

    @return left_R, right_R: The radius of curvature of left line and right line, respectevely. 
    '''
    
    # Extract left_poly_coeffs and right_poly_coeffs
    left_A = left_poly_coeffs[0]
    left_B = left_poly_coeffs[1]
    left_C = left_poly_coeffs[2]
    
    right_A = right_poly_coeffs[0]
    right_B = right_poly_coeffs[1]
    right_C = right_poly_coeffs[2]
    
    # Calculation of R_curve (radius of curvature)
    left_R = ((1 + (2*left_A*y_eval + left_B)**2)**1.5) / np.absolute(2*left_A)
    right_R = ((1 + (2*right_A*y_eval + right_B)**2)**1.5) / np.absolute(2*right_A)
    
    return left_R, right_R

def drawLanePoly(img, left_poly_coeffs, right_poly_coeffs, inv_Matrix):
    """
        Draw the area of the lane on the `img` of the road using the poly `left_poly_coeffs` and `right_poly_coeffs`.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    # Creating a blank image to draw into
    out_img = np.zeros_like(img).astype(np.uint8)

    # Extracting coefficients
    left_A = left_poly_coeffs[0]
    left_B = left_poly_coeffs[1]
    left_C = left_poly_coeffs[2]
    
    right_A = right_poly_coeffs[0]
    right_B = right_poly_coeffs[1]
    right_C = right_poly_coeffs[2]
    
    # Creating y-axis
    y = np.array(list(range(img.shape[0])))

    # Calculate points.
    left_fitX = left_A*y**2 + left_B*y + left_C
    right_fitX = right_A*y**2 + right_B*y + right_C

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_line_pts = np.array([np.transpose(np.vstack([left_fitX, y]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitX, y])))])
    pts = np.hstack((left_line_pts, right_line_pts))
    
    # Draw the polygon onto the warped blank image. We use a green color (0,255,0)
    cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(out_img, inv_Matrix, (img.shape[1], img.shape[0])) 
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)