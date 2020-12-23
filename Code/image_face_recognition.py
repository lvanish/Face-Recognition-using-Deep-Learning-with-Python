"""
@author: lvanish
"""
#importing required packages
import cv2
import face_recognition

#loading the image to detect
original_image = cv2.imread('images/testing/modi.jpg')


#load the sample images and get the 128 face encodings from them
modi_image = face_recognition.load_image_file('images/samples/modi_1.jpg')
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump_1.jpg')
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

#save the encodings and corresponding labels in seperate arrays in the same order
known_face_encodings = [modi_face_encoding,trump_face_encoding]
known_face_names = ["Narender Modi", "Donald Trump"]

#load the unknown image to recognize the faces in it
image_to_recognize = face_recognition.load_image_file('images/testing/modi.jpg')

#detect all faces in the image 
#arguments are image, no_of_times_to_upsample, model
all_face_locations = face_recognition.face_locations(image_to_recognize,model='hog')

#detect the face encodings for all the faces detected
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)

#printing number of faces detected in the array
print('There are {} no of faces in this image'.format(len(all_face_locations)))

#looping through the face locations and face embeddings
for current_face_locations,current_face_encoding in zip(all_face_locations,all_face_encodings):
    
    #splitting the tuple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_locations
    
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
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255), 1 )
    
    #display image
    cv2.imshow("Faces identified", original_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

