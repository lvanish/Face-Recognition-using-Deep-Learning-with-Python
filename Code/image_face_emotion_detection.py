"""
@author: lvanish
"""
#importing required packages
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

#loading the image to detect
image_to_detect = cv2.imread('images/testing/modi.jpg')

#load the model and load the weights
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')

#declaring list of emotions label
emotions_label = ('angry','disgust','fear','happy','sad','surprise','neutral')

#find all face locations using face_locations() fuction
#models can be hog or cnn
#optional argument number_of_times_to_upsample=1 higher and detect more faces
all_face_locations = face_recognition.face_locations(image_to_detect,model='hog')
#print(all_face_locations)

#printing number of faces detected in the array
print('There are {} no of faces in this image'.format(len(all_face_locations)))

#looping through the face locations
for index,current_face_locations in enumerate(all_face_locations):
    
    #splitting the tuple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_locations
    
    #printing the position of current face
    print('Found face {} at top: {}, right: {}, bottom: {}, left: {}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    
    #slicing the current face from main image
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    
    #draw rectangle around the face detected
    cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    
    #preprocess input, convert it to an image like as the data in dataset
    #convert to grayscale
    current_face_image = cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GRAY)
    
    #resize to 48x48 px size
    current_face_image = cv2.resize(current_face_image,(48,48))
    
    #convert the PIL image into a 3d numpy array 
    img_pixels = image.img_to_array(current_face_image)
    
    #expand the shapes of array into single row and multiple columns
    img_pixels = np.expand_dims(img_pixels,axis=0)
    
    #pixels are in range of [0,255]. Normalize all pixels in scale of [0,1]
    img_pixels /= 255
    
    
    #do prediction using model, get the prediction values of all 7 expressions
    exp_predictions = face_exp_model.predict(img_pixels)
    
    #find max indexed prediction value (0 till 7)
    max_index = np.argmax(exp_predictions[0])
    
    #get corresponding label from emotion_label
    emotion_label = emotions_label[max_index]
    
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255), 1)

#showing the current face with rectangle drawn
cv2.imshow("Image Face Emotions",image_to_detect)

cv2.waitKey(0)
cv2.destroyAllWindows()













"""
-> for displaying pictures
cv2.waitKey(0)
cv2.destroyAllWindows()
""" 