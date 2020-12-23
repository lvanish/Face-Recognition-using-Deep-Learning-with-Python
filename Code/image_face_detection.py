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
    #show each sliced face inside loop -> also use cv2.waitkey(0) and cv2.destroyAllWindows() to show image
    cv2.imshow("Face no "+str(index+1),current_face_image)

cv2.waitKey(0)
cv2.destroyAllWindows()













"""
-> for displaying pictures
cv2.waitKey(0)
cv2.destroyAllWindows()
"""