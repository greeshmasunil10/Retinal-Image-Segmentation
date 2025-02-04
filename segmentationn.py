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

global cf_matrix
global Y_train

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
#    choice = int(input("Enter test image number (21 to 40):" ))-21
    choice=0
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
    maxlist= []
#    maxlist 
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
#                print("...")
                           
    
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
    
    print(df.head())     
    
    X_features = df.drop(labels = ['Gray_chanel_labeled'],axis = 1)
    Y_labeled = df['Gray_chanel_labeled'].values
    
    df_len = len(df) 
    df_seperation_length = int(np.ceil(df_len/2)) 
    global Y_train
    X_train = X_features.head(df_seperation_length)
    print(X_train.head())
    X_test = X_features.tail(df_seperation_length - 1)
    Y_train, Y_test = np.split(Y_labeled, [df_seperation_length])
    print("train y",Y_train)
    
    #from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    #new_xtrain = sc.fit_transform(X_train)
    #new_x_test = sc.transform(X_test)
    ##
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 10, random_state = 42)
    
#    from sklearn.neighbors import KNeighborsClassifier
#    model = KNeighborsClassifier(n_neighbors=3)
    
    #
    #from sklearn.naive_bayes import MultinomialNB
    #model = MultinomialNB()
    
    model.fit(X_train, Y_train)
    
    prediction_test = model.predict(X_test)
    
    from sklearn.metrics import roc_curve 
    fper, tper, threshholds = roc_curve(Y_test,prediction_test)
    plt.plot(fper, tper, color = 'orange', label = 'ROC')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.show()           
    global cf_matrix
    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(Y_test, prediction_test)
    print(cf_matrix)
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


    print("Loading")
    
    
    import pandas as pd
#    import numpy as np
    
    data = df.head(df_seperation_length)
    testdata = df.tail(df_seperation_length - 1)
#    data = X_train
    
    #data['Gender'] = ['male','male','male','male','female','female','female','female']
    #data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
    #data['Weight'] = [180,190,170,165,100,150,130,150]
    #data['Foot_Size'] = [12,11,12,10,6,8,7,9]
    
    # View the data
#    data
    
    # Create an empty dataframe
    person = testdata
    
    # Create some feature values for this single row
    #person['Height'] = [5.75]..
    #person['Weight'] = [100]
    #person['Foot_Size'] = [15]
    
    # View the data 
#    print(person)
    # Number of males
#    n_male = data['Gray_chanel_labeled'][data['Gray_chanel_labeled'] == 255].count()
    n_female = data['Gray_chanel_labeled'][data['Gray_chanel_labeled'] == 255].count()
    
    # Number of males
#    n_female = Y_train[0][Y_train[0] == 0].count()
    n_male = data['Gray_chanel_labeled'][data['Gray_chanel_labeled'] == 0].count()
    
    # Total rows
    total_ppl = data['Gray_chanel_labeled'].count()
    
    # Number of males divided by the total rows
    P_male = n_male/total_ppl
    
    # Number of females divided by the total rows
    P_female = n_female/total_ppl
    
    
    # Group the data by gender and calculate the means of each feature
    data_means = data.groupby('Gray_chanel_labeled').mean()
    
    # View the values
#    data_means
    
    # Group the data by gender and calculate the variance of each feature
    data_variance = data.groupby('Gray_chanel_labeled').var()
    
    # View the values
#    data_variance
    
    # Means for male
    male_height_mean = data_means['green_Img'][data_variance.index == 0].values[0]
    male_weight_mean = data_means['Gabor2'][data_variance.index == 0].values[0]
    male_footsize_mean = data_means['Gabor3'][data_variance.index == 0].values[0]
    male_footsize_mean2 = data_means['Gabor4'][data_variance.index == 0].values[0]
    male_footsize_mean3= data_means['Gabor5'][data_variance.index == 0].values[0]
    
    # Variance for male
    male_height_variance = data_variance['green_Img'][data_variance.index == 0].values[0]
    male_weight_variance = data_variance['Gabor2'][data_variance.index == 0].values[0]
    male_footsize_variance = data_variance['Gabor3'][data_variance.index == 0].values[0]
    male_footsize_variance2 = data_variance['Gabor4'][data_variance.index == 0].values[0]
    male_footsize_variance3 = data_variance['Gabor5'][data_variance.index == 0].values[0]
    
    # Means for female
    female_height_mean = data_means['green_Img'][data_variance.index == 255].values[0]
    female_weight_mean = data_means['Gabor2'][data_variance.index == 255].values[0]
    female_footsize_mean = data_means['Gabor3'][data_variance.index == 255].values[0]
    female_footsize_mean2 = data_means['Gabor4'][data_variance.index == 255].values[0]
    female_footsize_mean3 = data_means['Gabor5'][data_variance.index == 255].values[0]
    
    # Variance for female
    female_height_variance = data_variance['green_Img'][data_variance.index == 255].values[0]
    female_weight_variance = data_variance['Gabor2'][data_variance.index == 255].values[0]
    female_footsize_variance = data_variance['Gabor3'][data_variance.index == 255].values[0]
    female_footsize_variance2 = data_variance['Gabor4'][data_variance.index == 255].values[0]
    female_footsize_variance3 = data_variance['Gabor5'][data_variance.index == 255].values[0]
    
    # Create a function that calculates p(x | y):
    def p_x_given_y(x, mean_y, variance_y):
    
        # Input the arguments into a probability density function
        p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
        
        # return p
        return p
    
    # Numerator of the posterior if the unclassified observation is a male
    p1=P_male * \
    p_x_given_y(person['green_Img'][0], male_height_mean, male_height_variance) * \
    p_x_given_y(person['Gabor2'][0], male_weight_mean, male_weight_variance) * \
    p_x_given_y(person['Gabor3'][0], male_footsize_mean, male_footsize_variance) * \
    p_x_given_y(person['Gabor4'][0], male_footsize_mean2, male_footsize_variance2) * \
    p_x_given_y(person['Gabor5'][0], male_footsize_mean3, male_footsize_variance3)
    # Numerator of the posterior if the unclassified observation is a female
    p2=P_female * \
    p_x_given_y(person['green_Img'][0], female_height_mean, female_height_variance) * \
    p_x_given_y(person['Gabor2'][0], female_weight_mean, female_weight_variance) * \
    p_x_given_y(person['Gabor3'][0], female_footsize_mean, female_footsize_variance)  * \
    p_x_given_y(person['Gabor4'][0], female_footsize_mean2, female_footsize_variance2)  * \
    p_x_given_y(person['Gabor5'][0], female_footsize_mean3, female_footsize_variance3)
    
    print(p1,p2)
    
    if(max(p1,p2)==p1):
        print("Person is a male")
    else:    
        print("Person is a female",)

controller()
