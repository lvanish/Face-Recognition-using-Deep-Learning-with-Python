"""
@author: lvanish
"""
#importing required packages
import cv2
import face_recognition

#loading the image to detect
image_to_detect = cv2.imread('images/testing/modi.jpg')

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
    #slice image array by positions inside the loop
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    
    #The 'AGE_GENDER_MODEL_MEAN_VALUES' calculated using the numpy.mean()
    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    #create blob of current face slice
    #parameters are image, scale, (size), (mean), RB swap
    current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (277,277), AGE_GENDER_MODEL_MEAN_VALUES, swapRB = False)
    
    #predicting Gender
    #declaring the labels
    gender_label_list = ['Male','Female']
    
    #declaring the file paths
    gender_protext = "dataset/gender_deploy.prototxt"
    gender_caffemodel = "dataset/gender_net.caffemodel"
    
    #creating the model
    gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
    
    #giving input to the model
    gender_cov_net.setInput(current_face_image_blob)
    
    #get predictions from the model
    gender_predictions = gender_cov_net.forward()
    
    #find the max value of predictions index
    #pass label to label array and get label text               
    gender = gender_label_list[gender_predictions[0].argmax()]
    
    
    #predicting Age
    #declaring the labels
    age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    #declaring the file paths
    age_protext = "dataset/age_deploy.prototxt"
    age_caffemodel = "dataset/age_net.caffemodel"
    
    #creating the model
    age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
    
    #giving input to the model
    age_cov_net.setInput(current_face_image_blob)
    
    #get predictions from the model
    age_predictions = age_cov_net.forward()
    
    #find the max value of predictions index
    #pass label to label array and get label text 
    age = age_label_list[age_predictions[0].argmax()]
    
    #draw rectangle around the face detected
    cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender + " " + age + " Years", (left_pos,bottom_pos+20), font, 0.5, (0,255,0), 1)

    #showing the current face with rectangle drawn
    cv2.imshow("Age and Gender",image_to_detect)

cv2.waitKey(0)
cv2.destroyAllWindows()











"""
-> for displaying pictures
cv2.waitKey(0)
cv2.destroyAllWindows()
""" 