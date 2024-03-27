import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


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

        p1k = face[33]
        p2k = face[160]
        p3k = face[158]
        p4k = face[133]
        p5k = face[153]
        p6k = face[144]

    img = cv2.flip(img, 1)
    cv2.imshow('Mirror', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
