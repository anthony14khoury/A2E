import socket

HOST = "127.0.0.1" # Standard loopback interface (localhost)
PORT = 65432 # Port to lisen on (non-priviledged ports are >= 1024)

# socket.socket creates a socket object
# AF_INET is the internet address family for IPv4
# SOCK_STREAM is the socket type for TCP
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    
     # bind is used to associate the socket with a specific network interface and port number
     s.bind((HOST, PORT))
     
     # listen enables a server to accept connections, it makes the server a "listening" socket
     s.listen()
     
     # accept blocks execution and waits for an incoming connection
     # when a client connects, it returns a new socket object representing the connection and a tuple holding the address of the client
     conn, addr = s.accept()
     with conn:
          print("Connected by ", str(addr))
          while True:
               
               # conn.recv reads whatever data the client sends and echoes it back using conn.sendall
               data = conn.recv(1024)
               if not data:
                    break
               conn.sendall(data)