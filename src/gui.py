# Interface graphique avec PyQt5
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import os
import cv2
from src.predict_fatigue import predict_fatigue
import tensorflow as tf

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "fatigue_detection_model.h5",
)


class FatigueApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def init_ui(self):
        self.setWindowTitle("Détection de Fatigue")
        self.setGeometry(100, 100, 800, 600)

        self.label = QtWidgets.QLabel("Analyse en cours", self)
        self.label.setGeometry(50, 50, 700, 50)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.start_button = QtWidgets.QPushButton("Démarrer", self)
        self.start_button.setGeometry(300, 150, 200, 50)
        self.start_button.clicked.connect(self.start_detection)

    def start_detection(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            is_fatigued = predict_fatigue(frame, self.model)
            text = "Fatigué" if is_fatigued else "Pas fatigué"
            self.label.setText(text)
            cv2.imshow("Détection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def start_gui():
    app = QtWidgets.QApplication(sys.argv)
    window = FatigueApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start_gui()