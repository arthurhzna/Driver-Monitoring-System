import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

def calculate_EAR(p1, p2, p3, p4, p5, p6):
    # Calculate distances
    d1 = np.linalg.norm(p2 - p6)
    d2 = np.linalg.norm(p3 - p5)
    d3 = np.linalg.norm(p1 - p4)
    # Calculate EAR
    ear = (d1 + d2) / (2 * d3)
    return ear

vid = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

idList = [61,39,0,269,291,181,17,405,33,160,158,133,144,153,362,385,387,263,380,373]

while True:

    success, img = vid.read()

    img, faces = detector.findFaceMesh(img, draw= False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img,face[id],2,(255,0,255),cv2.FILLED)

        p1k = np.array(face[33])
        p2k = np.array(face[160])
        p3k = np.array(face[158])
        p4k = np.array(face[133])
        p5k = np.array(face[153])
        p6k = np.array(face[144])

        ear_value = calculate_EAR(p1k, p2k, p3k, p4k, p5k, p6k)
        cv2.putText(img, f'EARKIRI: {ear_value:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        p1kN = np.array(face[263])
        p2kN = np.array(face[387])
        p3kN = np.array(face[385])
        p4kN = np.array(face[362])
        p5kN = np.array(face[380])
        p6kN = np.array(face[373])

    # img = cv2.flip(img, 1)
    cv2.imshow('Mirror', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
