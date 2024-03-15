import cv2

face_ref = cv2.CascadeClassifier('face_ref.xml')

camera = cv2.VideoCapture(0)

def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def drawer_box(frame, detections):
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def main():
    while True:
        _, frame = camera.read()
        if frame is None:
            break

        faces = face_detection(frame)
        frame_with_boxes = drawer_box(frame, faces)

        cv2.imshow('Face Detection', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
