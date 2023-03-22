import cv2
import time
import random

cap = cv2.VideoCapture(0)
if __name__ == "__main__":
    while True:
        num = 0
        val1 = time.time()
        while (num < 30):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            time.sleep(random.uniform(0.1, 0.2))
            cv2.imshow("Frame", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                quit()
            num += 1
        print(time.time() - val1)
        time.sleep(2)