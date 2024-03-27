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

def calculate_MAR(m1, m2, m3, m4, m5, m6, m7, m8):
    # Calculate distances
    d1 = np.linalg.norm(m2 - m8)
    d2 = np.linalg.norm(m3 - m7)
    d3 = np.linalg.norm(m4 - m6)
    d4 = np.linalg.norm(m1 - m5)
    # Calculate MAR
    mar = (d1 + d2 + d3) / (2 * d4)
    return mar

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

        ear_value_left = calculate_EAR(p1k, p2k, p3k, p4k, p5k, p6k)
        cv2.putText(img, f'EARKIRI: {ear_value_left:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        p1kN = np.array(face[263])
        p2kN = np.array(face[387])
        p3kN = np.array(face[385])
        p4kN = np.array(face[362])
        p5kN = np.array(face[380])
        p6kN = np.array(face[373])

        ear_value_right = calculate_EAR(p1kN, p2kN, p3kN, p4kN, p5kN, p6kN)
        cv2.putText(img, f'EARKANAN: {ear_value_right:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        m1 = np.array(face[61])
        m2 = np.array(face[39])
        m3 = np.array(face[0])
        m4 = np.array(face[269])
        m5 = np.array(face[291])
        m6 = np.array(face[405])
        m7 = np.array(face[17])
        m8 = np.array(face[181])

        mar_value = calculate_MAR(m1, m2, m3, m4, m5, m6, m7, m8)
        cv2.putText(img, f'MAR: {mar_value:.2f}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Cek apakah nilai ear_value_left atau ear_value_right kurang dari 0.3
        if ear_value_left < 0.25 or ear_value_right < 0.25:
            cv2.putText(img, "Mengantuk", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Tidak Mengantuk", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Mirror', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
