# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:31:01 2019

@author: Greeshma
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy import signal
import cv2
import matplotlib.image as mpimg
import time


def apply_wavelet_transform( sigma, theta, freq):
    
    filter_size = 10
    filter_half_size = int(filter_size / 2)
    [x, y] = np.meshgrid(range(-filter_half_size , filter_half_size+1), range(-filter_half_size , filter_half_size+1))
    
    pB = (-1) * (np.square(x) + np.square(y))
    pB = pB / (2 * sigma ** 2)
    g_sigma = (1 / (2 * np.pi * sigma ** 2)) * np.exp(pB)
    
    xcos = x * math.cos(theta)
    ysin = y * math.sin(theta)
    
    real_g = g_sigma * np.cos((2 * np.pi * freq) * (xcos + ysin))    
    img_g = g_sigma * np.sin((2 * np.pi * freq) * (xcos + ysin))
    
    return real_g, img_g
 
def controller():    
    choice = int(input("Enter test image number (21 to 40):" ))-21
    start= time.time()  
#    for i in range(20):
#        feature_extraction(i)
    feature_extraction(choice)
    """Containns few logical errors  but works fine"""
    # Read an Image
    end= time.time()
    print("\nTotal elapsed time:",round(end - start,1),"seconds") 

list_training_images = ['21_training.tif','22_training.tif','23_training.tif','24_training.tif','25_training.tif','26_training.tif','27_training.tif','28_training.tif','29_training.tif','30_training.tif','31_training.tif','32_training.tif','33_training.tif','34_training.tif','35_training.tif','36_training.tif','37_training.tif','38_training.tif','39_training.tif','40_training.tif']
list_labeled_images = ['21_manual1.gif','22_manual1.gif','23_manual1.gif','24_manual1.gif','25_manual1.gif','26_manual1.gif','27_manual1.gif','28_manual1.gif','29_manual1.gif','30_manual1.gif','31_manual1.gif','32_manual1.gif','33_manual1.gif','34_manual1.gif','35_manual1.gif','36_manual1.gif','37_manual1.gif','38_manual1.gif','39_manual1.gif','40_manual1.gif']

global dim

df = pd.DataFrame()
def convert(choice):
    global dim
    img = mpimg.imread(list_training_images[choice])
    width = int(img.shape[1] )
    height = int(img.shape[0])
    if width > height:
        width = height
    else:
        height = width
    dim = (width, height)
    Img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray_Img = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    [b,g,r] = cv2.split(Img)
    # Convert to green channel
    green_img = g
#    plt.imshow(green_img,cmap="gray")
#    plt.show()   
    cv2.waitKey(0)
    df['green_Img']  = green_img.flatten()
    return gray_Img


file_num=1
def feature_extraction(choice) :   
     
#    for it in range():
        
    gray_Img   = convert(choice)
        
    list_xy = [2, 3, 4, 5]
    list_theta = [np.pi/8, 2 * np.pi/8, 3 * np.pi/8, 4 * np.pi/8, 5 * np.pi/8, 6 * np.pi/8, 7 * np.pi/8 ]
    list_freq = [0.1, 0.2, 0.3]    
#    inverted_green = np.invert(green_img)
#    print("Loading..")
    for sigma_x in list_xy:
        
        maxlist = []
               
        for freq_z in list_freq:
                
            for angle_y in list_theta:
                   
                wavelet_output_real, wavelet_output_imag = apply_wavelet_transform( sigma_x, angle_y, freq_z)
                                  
                A = signal.fftconvolve(gray_Img, wavelet_output_real, mode = 'same')
                B = signal.fftconvolve(gray_Img, wavelet_output_imag, mode = 'same')
         
                A = A * A
                B = B * B
                C = A + B
                output_val = np.sqrt(C)
                output_val = (output_val - np.mean(output_val))/np.std(output_val)
                maxlist.append(output_val)
                print("...")
                           
    
        check= maxlist[0]
        for it in maxlist:
            maxout= np.maximum(check,it)
        out = np.zeros(maxout.shape, np.double)
                
        normalized = cv2.normalize(maxout, out, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        normalized_out = normalized * 255
        maxoutput_img = np.uint8(normalized_out)
        
#        list_max_det.append(maxout)
#        list_max_output_val.append(maxoutput_img)
        df['Gabor'+str(sigma_x)] = maxoutput_img.flatten()
  
        
    labeled_Image = mpimg.imread(list_labeled_images[choice])
    
    width = int(labeled_Image.shape[1] )
    height = int(labeled_Image.shape[0])
    
    if width > height:
        width = height
    else:
        height = width
    global dim    
    labeled_Image_resied = cv2.resize(labeled_Image, dim, interpolation = cv2.INTER_AREA)
    labeled_gray_Image = labeled_Image_resied
    # [b,g,r] = cv2.split(labeled_Image_resied)
    # Convert to green channel
    #labeled_green_image = g
    #plt.imshow(labeled_green_image)
    #labeled_Image_resied= np.invert(labeled_Image_resied)
    flatten_labeled_image = labeled_Image_resied.flatten()
    flatten_labeled_image[flatten_labeled_image > 0] = 1
    list_test_img = flatten_labeled_image.reshape(height,height)
#    plt.imshow(list_test_img, cmap = 'gray')
    
    flatten_labeled_image[flatten_labeled_image > 0] = 1
    df['Gray_chanel_labeled'] = flatten_labeled_image
         
    
    X_features = df.drop(labels = ['Gray_chanel_labeled'],axis = 1)
    Y_labeled = df['Gray_chanel_labeled'].values
    
    df_len = len(df) 
    df_seperation_length = int(np.ceil(df_len/2)) 
    
    X_train = X_features.head(df_seperation_length)
    X_test = X_features.tail(df_seperation_length - 1)
    Y_train, Y_test = np.split(Y_labeled, [df_seperation_length])
    
    #from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    #new_xtrain = sc.fit_transform(X_train)
    #new_x_test = sc.transform(X_test)
    ##
#    from sklearn.ensemble import RandomForestClassifier
#    model = RandomForestClassifier(n_estimators = 10, random_state = 42)
    
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_estimators = 10, random_state = 42)
    
    #
    #from sklearn.naive_bayes import MultinomialNB
    #model = MultinomialNB()
    
    model.fit(X_train, Y_train)
    
    prediction_test = model.predict(X_test)
    
    from sklearn.metrics import roc_curve 
    fper, tper, threshholds = roc_curve(Y_test,prediction_test)
#    plt.plot(fper, tper, color = 'orange', label = 'ROC')
    plt.show()           
    
    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(Y_test, prediction_test)
    prediction_test[prediction_test > 0] = 255
    
    
    Y_train.shape = (159613, 1)
    prediction_test.shape = (159612, 1)
    Y_train[Y_train > 0] = 255
    list_test_final = np.vstack((Y_train, prediction_test))
    list_test_img = list_test_final.reshape(height,height)
            
    #list_test_img = cv2.bitwise_not(list_test_img)
    plt.imshow(list_test_img, cmap = 'gray')
    Y_test[Y_test > 0] = 255
    #print(Y_test)
    #print(prediction_test)
    #for it in range(len(Y_test)):
    #    print(Y_test[it],":",prediction_test[it])
    from sklearn import metrics
    print('Acuracy is', metrics.accuracy_score(Y_test, prediction_test))
#    print('Precision is', metrics.precision_score(Y_test, prediction_test, average="micro"))
    filename = 'output\\' + str(metrics.accuracy_score(Y_test, prediction_test)) + '.png'
    plt.savefig(filename)
controller()
