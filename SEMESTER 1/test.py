import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("My Camera", frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    if key == ord('s'):
        cv2.imwrite("photo.jpg", frame)
        print("Photo saved!")

cap.release()
cv2.destroyAllWindows()