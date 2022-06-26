############################### IMPORTS #####################################
import numpy as np
import cv2
import math

############################ HAND DETECTION #################################
def HandDetection(img):
    
    # 1) closing to fill face gaps!!
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

    # 2) get contours
    contours,a = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 3) sort contours
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cntsSorted = cntsSorted[-2:]

    # 4) draw contours
    mask = np.zeros(img.shape)
    cv2.drawContours(mask, cntsSorted, -1 , 255, 1)

    # 5) get the bounding ellipse and calc the ration between ellipse area and contour area
    area_ratio = []
    cntsSorted = [cnt for cnt in cntsSorted if len(cnt)>=5]
    for cnt in cntsSorted:
        area = cv2.contourArea(cnt)
        ellipse = cv2.fitEllipse(cnt) 
        ellipse_Area = math.pi * ellipse[1][0] * ellipse[1][1]
        area_ratio.append(area/ellipse_Area)
        cv2.ellipse(mask, ellipse, 255, 3)

    # print(area_ratio)
    # 6) face has the max ratio
    if area_ratio:
        face = cntsSorted[np.argmax(area_ratio)]
        hand = cntsSorted[np.argmin(area_ratio)]
    else:
        return img

    # 7) remove the face 
    x,y,w,h = cv2.boundingRect(face)
    img[y:y+h,x:x+w] = 0
    # show_images([mask,img])

    
    # # 8 ) crop the image 
    x,y,w,h = cv2.boundingRect(hand)
    # roi=cv2.medianBlur(global_mask,9)
    # cropped=img[y:y+h,x:x+w]
    # print(y-int(h*0.2), y+h+int(h*0.2), x-int(w*0.2), x+w+int(w*0.2))
    # print(img.shape)
    y_min = max(0, y-int(h*0.2))
    x_min = max(0, x-int(w*0.2))

    y_max = min(y+h+int(h*0.2), img.shape[0])
    x_max = min(x+w+int(w*0.2), img.shape[1])

    cropped = img[y_min:y_max , x_min: x_max ]

    # cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

    return img, cropped