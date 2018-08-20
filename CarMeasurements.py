"""
==============
Edge operators
==============

Edge operators are used in image processing within edge detection algorithms.
They are discrete differentiation operators, computing an approximation of the
gradient of the image intensity function.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import scipy
import cv2

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt

#image = camera()
image =  np.array('C:\ActiveCodingProjects\CarMeasurements\TestFiles\car.jpg')
from PIL import Image

temp=Image.open('C:\ActiveCodingProjects\CarMeasurements\TestFiles\download.png')
#temp=temp.convert('1')      # Convert to black&white
A = np.array(temp)             # Creates an array, white pixels==True and black pixels==False
image=np.empty((A.shape[0],A.shape[1]),None)    #New array with same size as A

for i in range(len(A)):
    for j in range(len(A[i])):
        if A[i][j]==True:
            image[i][j]=0
        else:
            image[i][j]=1
            
print(image)
edge_roberts = roberts(image)
edge_sobel = sobel(image)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

from scipy import ndimage
edge_horizont = ndimage.sobel(image, 0)
edge_vertical = ndimage.sobel(image, 1)
magnitude = np.hypot(edge_horizont, edge_vertical)
magnitude
######################################################################
# Different operators compute different finite-difference approximations of
# the gradient. For example, the Scharr filter results in a less rotational
# variance than the Sobel filter that is in turn better than the Prewitt
# filter [1]_ [2]_ [3]_. The difference between the Prewitt and Sobel filters
# and the Scharr filter is illustrated below with an image that is the
# discretization of a rotation- invariant continuous function. The
# discrepancy between the Prewitt and Sobel filters, and the Scharr filter is
# stronger for regions of the image where the direction of the gradient is
# close to diagonal, and for regions with high spatial frequencies. For the
# example image the differences between the filter results are very small and
# the filter results are visually almost indistinguishable.
#
# .. [1] https://en.wikipedia.org/wiki/Sobel_operator#Alternative_operators
#
# .. [2] B. Jaehne, H. Scharr, and S. Koerkel. Principles of filter design.
#        In Handbook of Computer Vision and Applications. Academic Press,
#        1999.
#
# .. [3] https://en.wikipedia.org/wiki/Prewitt_operator

x, y = np.ogrid[:100, :100]
# Rotation-invariant image with different spatial frequencies
#img = np.exp(1j * np.hypot(x, y)**1.3 / 20.).real
from PIL import Image
import numpy

def image2pixelarray(filepath):
        """
        Parameters
        ----------
        filepath : str
            Path to an image file

        Returns
        -------
        list
            A list of lists which make it simple to access the greyscale value by
            im[y][x]
        """
        im = Image.open(filepath).convert('L')
        
        (width, height) = im.size
        greyscale_map = list(im.getdata())
        greyscale_map = numpy.array(greyscale_map)
        greyscale_map = greyscale_map.reshape((height, width))
        greyscale_map = np.where(greyscale_map > 200, 0, 255)
        greyscale_map = ndimage.binary_opening(greyscale_map)
        greyscale_map = ndimage.binary_closing(greyscale_map)
        greyscale_map = ndimage.binary_fill_holes(greyscale_map)
        
        return greyscale_map
img=image2pixelarray('C:\ActiveCodingProjects\CarMeasurements\TestFiles\car.jpg')
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)

diff_scharr_prewitt = edge_scharr - edge_prewitt
diff_scharr_sobel = edge_scharr - edge_sobel
max_diff = np.max(np.maximum(diff_scharr_prewitt, diff_scharr_sobel))

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_scharr, cmap=plt.cm.gray)
ax[1].set_title('Scharr Edge Detection')

ax[2].imshow(diff_scharr_prewitt, cmap=plt.cm.gray, vmax=max_diff)
ax[2].set_title('Scharr - Prewitt')

ax[3].imshow(diff_scharr_sobel, cmap=plt.cm.gray, vmax=max_diff)
ax[3].set_title('Scharr - Sobel')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
image = cv2.imread('C:\ActiveCodingProjects\CarMeasurements\TestFiles\carside.jpg')
        
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue
 
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
 
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
 
	# loop over the original points and draw them
	for (x, y) in box:
		 cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	#cv2.circle(orig, (int(box[0][0]), int(box[0][1])), 5, (0, 255, 0), -1)
	#cv2.circle(orig, (int(box[1][0]), int(box[1][1])), 5, (0, 255, 144), -1)
	#cv2.circle(orig, (int(box[2][0]), int(box[2][1])), 5, (0, 0, 255), -1)
	#cv2.circle(orig, (int(box[3][0]), int(box[3][1])), 5, (0, 0, 255), -1)
    
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2) 
    	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB/ 240
    
    # compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}cm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		2, (0, 255, 0), 2)
	cv2.putText(orig, "{:.1f}cm".format(dimB),
		(int(tlblX - 10), int(trbrY * 0.9)), cv2.FONT_HERSHEY_SIMPLEX,
		3, (255, 0, 255), 2)

	# show the output image
cv2.imshow("Image", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()