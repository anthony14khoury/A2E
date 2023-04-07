import time
import cv2
import socket
from multiprocessing import Process, Pipe
# from word_detection import add_spaces


def prediction(parent_conn):
    from parameters import Params, extract_hand_keypoints
    from keras.models import load_model
    import mediapipe as mp
    import numpy as np

    # Parameter Class
    params = Params()

    # ML Model
    try:
        model = load_model("./Models/96_full_tanh_model.h5")
    except:
        model = load_model("./A2E/CameraTracking/Models/128_26_15_model_tanh.h5")
    letters = params.LETTERS

    # Mediapipe Modules
    mp_hands = mp.solutions.hands


    # Socket Settings
    HOST = "10.136.49.55" # The server's hostname or IP address
    PORT = 4000 # The port used by the server



    # Use CV2 Functionality to create a Video Stream
    cap = cv2.VideoCapture(2)
    # cap = cv2.VideoCapture(0)
    while not cap.isOpened():
        pass
    print("Camera is connected")
    
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # connect to socket
           # s.connect((HOST, PORT))
            # Stay open while the camera is activated
            while cap.isOpened():

                FRAME_STORE = []
                IMAGE_STORE = []
                hands_count = 0
                t = time.time()
                parent_conn.send(1)
                for frame_num in range(params.FRAME_COUNT):
                    success, image = cap.read()
                    IMAGE_STORE.append(image)
                    time.sleep(.05)
                print("recording took: ", time.time()-t)
                parent_conn.send(2)
                # Loop through all of the frames
                t0 = time.time()
                for frame_num in range(params.FRAME_COUNT):
                    print(frame_num)
                    t1 = time.time()
                    # Capture a frame
                    image = IMAGE_STORE[frame_num]
                    #print("image read time:", time.time()-t1)
                    # Error Checking
                    # if not success:
                    #     print("Ignoring Empty Camera Frame")
                    #     continue

                    # Make detections
                    image.flags.writeable = False
                    t2 = time.time()
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                   # print("hands.process took: ", time.time()-t2)

                    if results.multi_handedness != None:
                        hands_count += len(results.multi_handedness)

                    # Draw the Detections on the hand (comment out for PI predictions)
                    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Don't need this if you are drawing landmarks
                    #image = draw_styled_landmarks(image, results)
                    t3 = time.time()
                    FRAME_STORE.append(extract_hand_keypoints(results))
                   # print("extract keypoints took:", time.time()-t3)
                    # Display Image
                    #cv2.imshow('WindowOutput', image)

                    # Breaking gracefully
                    #if cv2.waitKey(5) & 0xFF == ord('q'):
                        #cap.release()
                        #cv2.destroyAllWindows()
                       # quit()

                if hands_count > 0:
                    prediction = model.predict(np.expand_dims(FRAME_STORE, axis=0))
                    char_index = np.argmax(prediction)
                    confidence = round(prediction[0,char_index]*100, 1)
                    predicted_char = letters[char_index]
                   # s.send(predicted_char)
                    print(predicted_char, confidence)
                
                else:
                    print("Nothing")
                print("total prediction time:", time.time()-t0)
                print("sleeping")
                time.sleep(2)
                print("go")
                
def server(child_conn):
    from flask import Flask, Response
    
    app = Flask(__name__)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (320, 240)     
    # fontScale
    fontScale = 2       
    # Blue color in BGR
    color = (255, 0, 0)      
    # Line thickness of 2 px
    thickness = 2
    def gen_frames(cap):
        display = 2
        while True:
            time.sleep(0.05)
            success, frame = cap.read()
            if not success:
                break
            else:
                if(child_conn.poll()):
                    display = child_conn.recv()
                    
                if(display == 1):
                    image = cv2.putText(frame, 'Go', org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
                elif(display == 2):
                    image = cv2.putText(frame, 'Wait', org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
                    
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpeg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        cap = cv2.VideoCapture(2)
        return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    app.run(host=str(ip), port=5000, debug=False, threaded=True)
    

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    pred = Process(target=prediction, args=(parent_conn,))
    pred.start()
    # Listen on the public IP address of the server
    camera = Process(target=server, args=(child_conn,))
    camera.start()
    pred.join()
    camera.join()

    # if len(predicted_char) > 1:
            #     #ipdb.set_trace()
            #     curr_sen = np.concatenate((curr_sen,temp))
            #     curr_letters = ""
            #     curr_sen = np.append(curr_sen,predicted_char)
            #     temp = []
            # else:
            #     curr_letters += predicted_char
            #     answer,temp = add_spaces(curr_letters, curr_sen)  #Print out most likely placement of spaces, add dashes if none found

            # print(answer)
