import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import pickle
#import detect
import base64
import cv2
#import test
import uuid
import os
import urllib
import imutils
import predictor
# Load the jpg files into numpy arrays
def train(image_path,usn):
    try:
        try:
            encodings,usns = pickle.load(open("model.pkl","rb"))
        except Exception:
            encodings = []
            usns = []
        user = face_recognition.load_image_file(image_path);print("API")
        #use num_jitters to randomly distort face to make prediction more robust
        user_face_encoding = face_recognition.face_encodings(user, num_jitters=100)[0]
        encodings.append(user_face_encoding)
        usns.append(usn)
        pickle.dump((encodings,usns),open("model.pkl","wb"))
        return True
    except Exception as e:
        print(e)
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        # quit()
        return False

def predict_from_encoding(face_encoding):
    # frame = cv2.imread(image_path)
    celebs_folder = os.path.join(os.getcwd(),'faces')
    encodings,usns = pickle.load(open("model.pkl","rb"))
    # cv2.imshow("ss",frame)
    # cv2.waitKey(1)


# See if the face is a match for the known face(s)

    distances = face_recognition.face_distance(encodings, face_encoding)
    matches = face_recognition.compare_faces(encodings, face_encoding, tolerance = 0.6)
    try:
        best_match_index = np.argmin(distances)
        
        first_match_index = best_match_index
        print("Index = ",best_match_index)
        # matching_indices = np.argwhere(distances < 0.63).flatten()

    except Exception as e:
        print('Error',e)
        first_match_index = -999

    name = "Unknown"
    print(matches)
    # If a match was found in , just use the first one.
    if first_match_index != -999 and matches[first_match_index]:
        name = usns[first_match_index]
        print(name)

    print(name)
    return name


def predict(image_path):
    try:

        encodings,usns = pickle.load(open("model.pkl","rb"))
        # user = face_recognition.load_image_file(image_path)
        unknown_image = face_recognition.load_image_file(image_path)
        pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
        draw = ImageDraw.Draw(pil_image)

# Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)

            matches = face_recognition.compare_faces(encodings, face_encoding, tolerance = 0.6)
            x,y,w,h = left,top,right-left,bottom-top
            face = unknown_image[y-20:y+h+20,x-20:x+w+20]
            age, gender = predictor.predict(face)
            print("********")
            print(f"{age}, {gender}")
            print("********")

            distances = face_recognition.face_distance(encodings, face_encoding)
            try:
                best_match_index = np.argmin(distances)
                
                first_match_index = best_match_index
                print("Index = ",best_match_index)
                # matching_indices = np.argwhere(distances < 0.63).flatten()

            except Exception as e:
                print('Error',e)
                first_match_index = -999

            name = "Unknown"
            print(matches)
            # If a match was found in , just use the first one.
            if first_match_index != -999 and matches[first_match_index]:
                name = usns[first_match_index]
                print(name)

            print(name)

            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name+f"\nA-{age[0][0]},G-{gender}")
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name+f"\nA-{age[0][0]:.2f},G-{gender}", fill=(255, 255, 255, 255))
            # draw.text((left + 6, bottom - text_height - 10), f"A-{age},G-{gender}", fill=(255, 255, 255, 255))

                
        del draw
        pil_image.save("output.png")

        with open("output.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return urllib.parse.quote(encoded_string),name
    except Exception as e:
        print('error',e)
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        # quit()
        return "","Unknown"

def recognize_faces_from_frame(frame, name):
    # celebs_folder = os.path.join(os.getcwd(),'faces')
    encodings,usns = pickle.load(open("model.pkl","rb"))
    # cv2.imshow("ss",frame)
    # cv2.waitKey(1)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
# See if the face is a match for the known face(s)

        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance = 0.6)
        distances = face_recognition.face_distance(encodings, face_encoding)
        try:
            best_match_index = np.argmin(distances)
            if matches[best_match_index]:
                first_match_index = best_match_index
            else:
                first_match_index = -999

            # first_match_index = np.argwhere(distances < 0.63).flatten()[0]

        except Exception as e:
            print(e)
            first_match_index = -999

        nameP = "Unknown"


        # If a match was found in known_face_encodings, just use the first one.
        if first_match_index != -999:
            nameP = usns[first_match_index]

        if nameP:
            print(nameP)
            x,y,w,h = left,top,right-left,bottom-top
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255), 1)
            cv2.putText(frame, nameP, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    # cv2.imshow("frame",frame)
    # cv2.waitKey(2)
    return frame, len(face_locations)
            # cv2.imwrite(os.path.join(celebs_folder,name,str(uuid.uuid1())+".jpg"),frame)


def face_extractor(video_file, name):
    cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            save_faces_from_frame(frame, name)
        
    # When everything done, release the video capture object
    cap.release()
    return "success"

# # Display the resulting image
# pil_image.show()
if __name__ == "__main__":
    # face_extractor("survillance.mp4", "amitabh")
    print(predict("anil.jpg"))
