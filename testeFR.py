
import cv2
import face_recognition as fr
import mediapipe as mp


webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection.FaceDetection()
desenhador = mp.solutions.drawing_utils

imgMatheus = fr.load_image_file('matheus2.jpg')
imgMatheus = cv2.cvtColor(imgMatheus, cv2.COLOR_BGR2RGB)
#imgMatheusTeste = fr.load_image_file('MatheusTeste.jpg')
#imgMatheusTeste = cv2.cvtColor(imgMatheusTeste, cv2.COLOR_BGR2RGB)
facelocMatheus = fr.face_locations(imgMatheus)[0]
encodeMatheus = fr.face_encodings(imgMatheus)[0]

imgMichele = fr.load_image_file('michele.jpg')
imgMichele = cv2.cvtColor(imgMichele, cv2.COLOR_BGR2RGB)
#imgMatheusTeste = fr.load_image_file('MatheusTeste.jpg')
#imgMatheusTeste = cv2.cvtColor(imgMatheusTeste, cv2.COLOR_BGR2RGB)
facelocMichele = fr.face_locations(imgMichele)[0]
encodeMichele = fr.face_encodings(imgMichele)[0]


#facelocMatheusTeste = fr.face_locations(imgMatheusTeste)[0]


#encodeMatheusTeste = fr.face_encodings(imgMatheusTeste)[0]


#comparacao = fr.compare_faces([encodeMatheus], encodeMatheusTeste)
# print(comparacao)
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

cv2.waitKey(0)
