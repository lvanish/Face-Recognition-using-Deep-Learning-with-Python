"""
@author: lvanish
"""
#importing required packages
import cv2
import face_recognition

#get the webcam using VideoCapture(0), 0 is for default camera
webcam_video_stream = cv2.VideoCapture(0)

#initialize the array variable to hold all face locations in the frame
all_face_locations = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image, first variable returns boolean value second variable returns image at exact point where the system is executing 
    ret,current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    
    #find all face locations using face_locations() fuction
    #models can be hog or cnn
    #optional argument number_of_times_to_upsample=1 higher and detect more faces
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    
    #looping through the face locations
    for index,current_face_locations in enumerate(all_face_locations):
        #splitting the tuple to get the four position values
        top_pos,right_pos,bottom_pos,left_pos = current_face_locations
        #change the position magnitude to fit the actual size of video frame
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        #printing the position of current face                              
        print('Found face {} at top: {}, right: {}, bottom: {}, left: {}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        
        #Extracting the face from the frame, blur it, paste it back to the frame
        #slicing the current face from main image
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
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
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, gender + " " + age + " Years", (left_pos,bottom_pos+20), font, 0.5, (0,255,0), 1)
    
    #showing the current face with rectangle drawn
    cv2.imshow("webcam video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to terminate 
        break

webcam_video_stream.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    