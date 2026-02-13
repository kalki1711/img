import os
import glob
import cv2
import face_recognition
import numpy as np
class ImprovedFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.face_recognition_model = "cnn" 
        self.threshold = 0.6  
    def load_encoding_images_from_dataset(self, dataset_path):
        person_folders = glob.glob(os.path.join(dataset_path, "*"))
        print("{} persons found in the dataset.".format(len(person_folders)))
        for person_folder in person_folders:
            person_name = os.path.basename(person_folder)
            print("Loading images for person: {}".format(person_name))
            image_files = glob.glob(os.path.join(person_folder, "*.jpg")) 
            if not image_files:
                print("No images found for {}".format(person_name))
                continue
            for img_path in image_files:
                img = cv2.imread(img_path)
                img_encoding = face_recognition.face_encodings(rgb_img)
                if not img_encoding:
                    print("No face found in {}".format(img_path))
                    continue
                self.known_face_encodings.append(img_encoding[0]) 
                self.known_face_names.append(person_name)
        print("Encoding images loaded")
    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        if self.face_recognition_model == "cnn":
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        else:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        if len(face_encodings) == 0:
            return np.array([]), []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.threshold)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) == 0:
                continue
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
if __name__ == "__main__":
    sfr = ImprovedFacerec()
    sfr.load_encoding_images_from_dataset("data/")  
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

