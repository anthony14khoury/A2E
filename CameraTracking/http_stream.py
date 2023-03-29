import cv2
from flask import Flask, Response
import time
app = Flask(__name__)

def gen_frames(cap):
    while True:
        time.sleep(0.05)
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(2)
    return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Listen on the public IP address of the server
    app.run(host='10.192.184.58', port=5000, debug=True)