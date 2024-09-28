import sys
import cv2
import numpy as np
import face_recognition
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PIL import Image, ImageDraw, ImageFont
from utils import faces_distance_cosine
import os
import pickle
import time


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.known_face_names = []
        self.known_face_encodings = []
        self.load_known_faces()
        self.show_landmarks = False  # 初始化面部关键点标注状态


    def initUI(self):
        self.setWindowTitle('Face Recognition App')
        self.setGeometry(100, 100, 1280, 720)
        self.setWindowIcon(QIcon('D:/PycharmProjects/faceRecognition/images/icon.png'))
        self.setStyleSheet("background-color: #f0f0f0;")

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 720)
        self.video_label.setStyleSheet("border: 2px solid black;")

        self.start_button = QPushButton('Start', self)
        self.start_button.setFont(QFont('Arial', 14))
        self.start_button.setStyleSheet(
            "background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px;"
        )
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.setFont(QFont('Arial', 14))
        self.stop_button.setStyleSheet(
            "background-color: #f44336; color: white; border-radius: 10px; padding: 10px;"
        )
        self.stop_button.clicked.connect(self.stop_video)

        self.landmarks_button = QPushButton('Toggle Landmarks', self)
        self.landmarks_button.setFont(QFont('Arial', 14))
        self.landmarks_button.setStyleSheet(
            "background-color: #2196F3; color: white; border-radius: 10px; padding: 10px;"
        )
        self.landmarks_button.clicked.connect(self.toggle_landmarks)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.landmarks_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture()

    def start_video(self):
        self.cap.open(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1900)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.timer.start(1)

    def stop_video(self):
        self.timer.stop()
        self.cap.release()
        self.video_label.clear()

    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks  # 切换面部关键点标注状态

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.recognize_faces(frame)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(convert_to_qt_format))

    def recognize_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 缩放， 加速处理
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # 转化到连续内存空间，加速处理
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.43)
            matches2 = faces_distance_cosine(self.known_face_encodings, face_encoding)

            name = "疑似未录入人员"
            tolerance1 = 0.4
            tolerance2 = 0.939
            epsilon = 0.3
            most_match1 = min(matches)
            most_match2 = max(matches2)
            if most_match1 <= tolerance1 and most_match2 >= tolerance2:
                first_match_index = matches.index(most_match1)
                first_match_index2 = matches2.index(most_match2)
                prob = np.random.random()

                if prob >= epsilon:
                    name = self.known_face_names[first_match_index]
                else:
                    name = self.known_face_names[first_match_index2]

            elif most_match1 > tolerance1 and most_match2 < tolerance2:
                name = "未录入人员"

            face_names.append(name)

        for (top, right, bottom, left), name, face_landmarks in zip(face_locations, face_names, face_landmarks_list):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == '未录入人员':
                color = (0, 0, 255)
            elif name == '疑似未录入人员':
                color = (0, 165, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            frame = self.put_cn_text(frame, name, (left + 6, bottom - 24), 20, (255, 255, 255))

            if  name == "未录入人员":
                self.save_unknown_face(frame, top, right, bottom, left)

            if self.show_landmarks:  # 根据状态决定是否标注面部关键点
                for feature in face_landmarks.keys():
                    for point in face_landmarks[feature]:
                        point = (point[0] * 4, point[1] * 4)
                        cv2.circle(frame, point, 1, (0, 255, 0), -1)

        return frame

    def load_known_faces(self):
        if os.path.exists('known_faces.pkl'):
            with open('known_faces.pkl', 'rb') as f:
                known_faces = pickle.load(f)
                self.known_face_names = known_faces['names']
                self.known_face_encodings = known_faces['encodings']
        else:
            path = os.path.abspath('images')
            print('Encoding Begins')
            start = time.perf_counter()
            for photo in os.listdir(path):
                self.add_photo(name=str(photo).split('.')[-2], filename=path + f'\\{photo}')
            end = time.perf_counter()
            print('Encoding ends')

            duration = end - start

            print(f'Duration: {duration}s')

            with open('known_faces.pkl', 'wb') as f:
                known_faces = {
                    'names': self.known_face_names,
                    'encodings': self.known_face_encodings
                }
                pickle.dump(known_faces, f)

    def add_photo(self, name, filename):
        image = face_recognition.load_image_file(filename)
        face_encoding = face_recognition.face_encodings(image)

        if len(face_encoding) != 1:
            print(f"Failed to encode face for {name} from {filename}")
            return False

        self.known_face_encodings.insert(0, face_encoding[0])
        self.known_face_names.insert(0, name)
        print(f"Added {name} from {filename}")
        return True

    def put_cn_text(self, image, strs, local, sizes, colour):
        cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)
        font = ImageFont.truetype("./simhei.ttf", sizes, encoding="utf-8")
        draw.text(local, strs, colour, font=font)
        return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    def save_unknown_face(self, frame, top, right, bottom, left):
        path = os.path.abspath('intruders')
        roi = frame[top:bottom, left:right]
        filename = f"unknown_{int(time.time())}.jpg"
        save_path = path + filename
        cv2.imwrite(save_path, roi)
        print(f"Saved unknown face to {filename}")


def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
