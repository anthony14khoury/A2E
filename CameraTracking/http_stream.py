import cv2
from flask import Flask, Response
import time
import socket
app = Flask(__name__)

def gen_frames(cap):
    while True:
        time.sleep(0.05)
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(2)
    return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    # Listen on the public IP address of the server
    app.run(host=str(ip), port=5000, debug=True)
