import face_recognition
import cv2
import numpy as np
import os

known_faces= [] 
known_names = []

stored_faces_path = PATH_TO_IMAGES

for filename in os.listdir(stored_faces_path): #iterate through all image files in faces folder
    if filename.endswith('.jpg') or filename.endswith('.png'):
        file_path = os.path.join(stored_faces_path, filename)
        with open(file_path, 'rb') as f:
            img = cv2.imread(file_path)
            if img is not None:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                known_faces.append(face_recognition.face_encodings(rgb_img)[0]) #encoding images in images folder after being facially recognized
                
                split_file_name = os.path.splitext(filename) 
                known_names.append(split_file_name[0]) #storing file name as name of person
            else:
                print(f"Error reading image: {file_path}")

video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

while True:
   
    ret, frame = video_capture.read() #Get an image frame

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  #Use 1/4 sized frame

    rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB) #convert to RGB
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame)  # Find all the faces and face encodings in the current frame of video

    face_names = []
    for face_encoding in face_encodings:

        result = face_recognition.compare_faces(known_faces, face_encoding) #check any of the known faces match the one in the video
        name = "Unknown"

        true_index = next((index for index, value in enumerate(result) if value), -1)

        if (true_index != -1):
            name = known_names[true_index] #find the index of where there was a match and assign its name
            
        face_names.append(name) # put it in list to display rectangles with names

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2) # Draw a rectangle around the faces in the video

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left, bottom + 30), font, 1.0, (255, 255, 255), 1) #write name under bottom left of rectangle

    cv2.imshow('Video', frame) #show webcam video with boxes around faces with names

    k = cv2.waitKey(30) & 0xff #wait for key to be pressed within a time frame
    if k == 27: #if key is Escape key break loop and terminate program
        break

video_capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
