"""
@author: lvanish
"""
#importing required packages
import cv2
import face_recognition

#get the webcam using VideoCapture(0), 0 is for default camera
webcam_video_stream = cv2.VideoCapture(0)

#load the sample images and get the 128 face encodings from them
modi_image = face_recognition.load_image_file('images/samples/modi_1.jpg')
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump_1.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

lvanish_image = face_recognition.load_image_file('images/samples/lvanish_1.jpg')
lvanish_face_encoding = face_recognition.face_encodings(lvanish_image)[0]

#save the encodings and corresponding labels in seperate arrays in the same order
known_face_encodings = [modi_face_encoding,trump_face_encoding,lvanish_face_encoding]
known_face_names = ["Narender Modi", "Donald Trump", "Lvanish"]

#initialize the array to hold all face locations, encodings and labels in the frame
all_face_locations = []
all_face_encodings = []
all_face_names = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image, first variable returns boolean value second variable returns image at exact point where the system is executing 
    ret,current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    
    #find all face locations using face_locations() fuction
    #models can be hog or cnn
    #optional argument number_of_times_to_upsample=1 higher and detect more faces
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
    
    #detect the face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)

    #looping through the face locations and face embeddings
    for current_face_locations,current_face_encoding in zip(all_face_locations,all_face_encodings):
        
        #splitting the tuple to get the four position values
        top_pos,right_pos,bottom_pos,left_pos = current_face_locations
        
        #change the position magnitude to fit the actual size of video frame
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        
        #string to hold the label
        name_of_person = 'Unknown face'
        
        #check if all_matches have at least one item
        # if yes, get the index number of face that is located in the first index of all_matches
        # get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
            
        #draw rectangle around face
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
            
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255), 1 )
        
    #showing the video
    cv2.imshow("webcam video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to terminate
        break

webcam_video_stream.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    