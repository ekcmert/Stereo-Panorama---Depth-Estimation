import cv2
import os

import imutils
import numpy as np

def trim(stitched):
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                  cv2.BORDER_CONSTANT, (0, 0, 0))

    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")

    cv2.imshow(folder, mask)
    cv2.waitKey(0)

    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # create two copies of the mask: one to serve as our actual
    # minimum rectangular region and another to serve as a counter
    # for how many pixels need to be removed to form the minimum
    # rectangular region
    minRect = mask.copy()
    sub = mask.copy()

    cv2.imshow(folder, minRect)
    cv2.waitKey(0)

    # keep looping until there are no non-zero pixels left in the
    # subtracted image
    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

        cv2.imshow(folder, minRect)
        cv2.waitKey(0)
        cv2.imshow(folder, sub)
        cv2.waitKey(0)

    # find contours in the minimum rectangular mask and then
    # extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    # use the bounding box coordinates to extract the our final
    # stitched image
    stitched = stitched[y:y + h, x:x + w]
    return stitched



mainfolder = 'images'
myfolder = os.listdir(mainfolder)
print(myfolder)

for folder in myfolder:
    path = mainfolder + '/' +folder
    images = []
    mylist = os.listdir(path)
    print(f'total number of images detected {len(mylist)}')
    stitcher = cv2.Stitcher.create()
    for imgN in mylist:
        current_image = cv2.imread(f'{path}/{imgN}')
        current_image = np.where(current_image == 0, 1, current_image)
        current_image = cv2.resize(current_image,(700,700),None)
        images.append(current_image)
        print(len(images))
        if(len(images)>1):
            (status,result) = stitcher.stitch(images)
            if(status == cv2.STITCHER_OK):
                print('panorama done successfully')
                cv2.imshow(folder,result)
                cv2.waitKey(10)
            else:
                print ('panorama is unsuccessfull')
    rect = result
    rect = trim(rect)
    cv2.imwrite(f'{folder}.jpg', result)
    cv2.imwrite(f'{folder}rect.jpg', rect)


img = cv2.imread(f'{folder}.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect corners
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

# Find the perpective transform matrix
src_points = np.float32([corners[i] for i in range(corners.shape[0])])
dst_points = np.float32([[i, j] for i in range(corners.shape[0]) for j in range(corners.shape[1])])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Warp the image
result = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
cv2.imshow("Result", result)

