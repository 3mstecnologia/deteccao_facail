import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection.FaceDetection()
desenhador = mp.solutions.drawing_utils


while True:
    conectado, frame = webcam.read()
    if not conectado:
        print("Não foi possível conectar a webcam")
        break

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    resultados = solucao_reconhecimento_rosto.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if resultados.detections:
        for rosto in resultados.detections:
            desenhador.draw_detection(frame, rosto)

    cv2.imshow("Detector de rostos", frame)

    if cv2.waitKey(1) == ord("q"):
        break
