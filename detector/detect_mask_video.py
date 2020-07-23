# python detect_mask_video.py
import time
import cv2
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import subprocess as sp


class detect:
    def __init__(self, prototxtPath, weightsPath):
        self.prototxtPath = prototxtPath
        self.weightsPath = weightsPath
        # self.videoPath = videoPath
        self.predict()

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
        return locs, preds

    def predict(self):
        faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
        print("Loading face mask detector model...")
        maskNet = load_model('detector/mask_detector.model')
        print("Starting video stream...")

        cap = cv2.VideoCapture('videoplayback.mp4')
        filename = r"C:\\Users\\yeshw\\Documents\\opencvvid\\out.avi"

        ffmpeg = 'FFMPEG'
        dimension = '{}x{}'.format(600, 400)
        f_format = 'bgr24'
        fps = str(cap.get(cv2.CAP_PROP_FPS))

        command = [ffmpeg,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', dimension,
                   '-pix_fmt', 'bgr24',
                   '-r', fps,
                   '-i', '-',
                   '-an',
                   '-vcodec', 'mpeg4',
                   '-b:v', '5000k',
                   filename]

        proc = sp.Popen(command)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            proc.stdin.write(frame.tostring())

        cap.release()
        proc.stdin.close()
        proc.stderr.close()
        proc.wait()

        # vs = VideoStream(src=0).start()
        # time.sleep(2.0)

        # while True:
        #     frame = vs.read()
        #     frame = imutils.resize(frame, width=500)
        #     (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)
        #     for (box, pred) in zip(locs, preds):
        #         (startX, startY, endX, endY) = box
        #         (mask, withoutMask) = pred
        #
        #         label = "Mask" if mask > withoutMask else "No_Mask"
        #         color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        #         label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        #
        #         cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        #         cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        #
        #     cv2.imshow("FaceMaskDetector", frame)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord("q"):
        #         break
        #
        # cv2.destroyAllWindows()
        # vs.stop()
