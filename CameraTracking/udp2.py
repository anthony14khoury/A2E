import cv2
import numpy as np
import socket
import base64
import imutils
# Set up UDP socket
BUFF_SIZE = 65536
UDP_IP = "10.136.112.237" # Change this to your server's IP address
UDP_PORT = 5000 # Choose a port number
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)

# Set up video capture
cap = cv2.VideoCapture(0) # Use default webcam (change to a different number if you have multiple webcams)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = imutils.resize(frame,width=15)
    encoded,frame = cv2.imencode('.jpg',frame)
    # Convert frame to a string of bytes
   # message = base64.b64encode(frame)

    data = np.array(frame)
    string_data = data.tostring()

    # Send frame to UDP server
    sock.sendto(string_data, (UDP_IP, UDP_PORT))

# Release the capture and close the socket
cap.release()
sock.close()