import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

################################## SHOW IMAGE ############################################
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


################################## SevenHuMoments ############################################

def seven_hu_moments(img):
    moments = cv2.moments(img)
    huMoments = cv2.HuMoments(moments)
    for i in range(0,7):
        if(huMoments[i] != 0):
            huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

    return huMoments


def hu_moments(img):
    M00 = 0
    M10 = 0
    M01 = 0

    M11 = 0
    M20 = 0
    M02 = 0

    M30 = 0
    M03 = 0

    M21 = 0
    M12 = 0

    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):
            M00 += img[y][x]
            M10 += x * img[y][x]
            M01 += y * img[y][x]
    x_bar = M10/M00
    y_bar = M01/M00

    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):

            M11 += (x-x_bar)*(y-y_bar)* img[y][x]

            M20 += ((x-x_bar)**2) * img[y][x]
            M02 += ((y-y_bar)**2) * img[y][x]

            M30 += ( ((x-x_bar)**3) * img[y][x] ).astype('float64')
            M03 += ( ((y-y_bar)**3) * img[y][x] ).astype('float64')

            M21 += ( ((x-x_bar)**2) * (y-y_bar) * img[y][x]).astype('float64')
            M12 += ( ((y-y_bar)**2) * (x-x_bar) * img[y][x]).astype('float64')

    M = np.zeros((7))
    M[0] = (M20/(M00**2)) + (M02/(M00**2))

    M[1] = ((M20/(M00**2) - M02/(M00**2))**2) + 4* (M11/(M00**2))**2

    M[2] = (( M30/(M00**2.5))-(3* M12/(M00**2.5)) )**2 + ((3*M21/(M00**2.5))-(M03/(M00**2.5)))**2

    M[3] = (M30/(M00**2.5) + M12/(M00**2.5))**2 + (M21/(M00**2.5) + M03/(M00**2.5))**2

    M[4] = (M30/(M00**2.5) - 3*M12/(M00**2.5))*(M30/(M00**2.5)+M12/(M00**2.5))*((M30/(M00**2.5)+M12/(M00**2.5))**2 - 3*(M21/(M00**2.5)+M03/(M00**2.5))**2) + (3*M21/(M00**2.5) -M03/(M00**2.5))*(M21/(M00**2.5) + M03/(M00**2.5))*(3*(M30/(M00**2.5)+M12/(M00**2.5))**2 - (M21/(M00**2.5)+M03/(M00**2.5))**2)
    M[5] = (M20/(M00**2)-M02/(M00**2))* ( (M30/(M00**2.5)+M12/(M00**2.5))**2 - (M21/(M00**2.5) + M03/(M00**2.5))**2) + 4 * (M11/(M00**2))*(M30/(M00**2.5) + 3*M12/(M00**2.5))*(M21/(M00**2.5)+M03/(M00**2.5))
    M[6] = (3*M21/(M00**2.5) - M03/(M00**2.5))*(M30/(M00**2.5) + M12/(M00**2.5)) *((M30/(M00**2.5)+M12/(M00**2.5))**2 - 3*(M21/(M00**2.5)+M03/(M00**2.5))**2) - (M30/(M00**2.5) - 3*M12/(M00**2.5))*(M21/(M00**2.5)+ M03/(M00**2.5))*(3*(M30/(M00**2.5)+M12/(M00**2.5))**2 - (M21/(M00**2.5)+M03/(M00**2.5))**2)
    
    return M
    # moments = cv2.moments(closing.astype(np.uint8))
    # huMoments = cv2.HuMoments(moments)