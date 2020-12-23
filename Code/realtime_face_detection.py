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
        #draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    
    #showing the current face with rectangle drawn
    cv2.imshow("webcam video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    