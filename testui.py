import sys
from os import path
import os

import cv2
import numpy as np
from darkflow.net.build import TFNet
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets  import QLineEdit,QPlainTextEdit,QMessageBox
import pyttsx3

class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)
    
    def stop_recording(self):
        self.timer.stop()


    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class SignDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        # Create textbox
        #self.text_screen = QPlainTextEdit(self)
        self._red = (0, 0, 255)
        self._width = 5
        self._min_size = (30, 30)
        self.sample = 0
        self.loaded = False
        self.lastletter = ""
        self.timeofletter = 0
        self.text_screen = QPlainTextEdit(self)
        self.tosay = []

    def load(self):
        self. options = {
                'model': 'yolo-sign.cfg',
                'load': 'yolo-sign_33000.weights',
                'threshold': 0.2,
            }
        self.tfnet = TFNet(self.options)
        self.colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
        self.loaded = True
        self.tosay = []

    def stop(self):
        self.tfnet = ""
        self.loaded = False
        self.speak()
    
    def clean(self):
        self.setText("")
        self.tosay = []
        self.text_screen = QPlainTextEdit(self)

    def image_data_slot(self, image_data):
        if self.loaded:
            results = self.tfnet.return_predict(image_data)
            for color, result in zip(self.colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence * 100)
                image_data = cv2.rectangle(image_data, tl, br, color, 5)
                image_data = cv2.putText(
                image_data, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            if results:
                self.text_screen.move(10,10)
                self.text_screen.resize(200,100)
                if self.lastletter != result['label']:
                    self.timeofletter= 0
                    self.lastletter  = result['label']
                if self.lastletter == result['label'] and self.timeofletter == 10:
                    if result['label']== "ESPACIO":
                        result['label']= " "
                    if result['label']== "PARA":
                        result['label']= ""
                        self.stop()
                    self.tosay.append(result['label'])
                    self.text_screen.insertPlainText(result['label'])
                    self.timeofletter= 0
                    self.lastletter  = ""
                else:
                    self.timeofletter += 1
                    self.lastletter = result['label']
        self.image = self.get_qimage(image_data)
            #self.text_screen.setText(''.join(self.tosay))
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        
        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()
        
    def speak(self):
        word = ''.join(self.tosay)
        sufix1 = 'SION'
        sufix2 = 'CION'
        if word.find(sufix1):
            word=word.replace(sufix1,"SIÓN")
        if word.find(sufix2):
            word=word.replace(sufix2,"CIÓN")
        myCmd = 'gtts-cli --lang es \''+ word +'\' | play -t mp3 -'
        os.system(myCmd)







class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_detection_widget = SignDetectionWidget()
        self.record_video = RecordVideo()
        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.face_detection_widget)   
        self.load = QtWidgets.QPushButton('Load')
        layout.addWidget(self.load)
        self.load.clicked.connect(self.face_detection_widget.load)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.record_video.start_recording)
        self.run_stop = QtWidgets.QPushButton('Stop')
        layout.addWidget(self.run_stop)
        self.run_stop.clicked.connect(self.record_video.stop_recording)
        self.speak = QtWidgets.QPushButton('Speak')
        layout.addWidget(self.speak)
        self.speak.clicked.connect(self.face_detection_widget.speak)
        self.clean = QtWidgets.QPushButton('Clean')
        layout.addWidget(self.clean)
        self.speak.clicked.connect(self.face_detection_widget.clean)
        self.setLayout(layout)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget()
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

