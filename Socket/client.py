import socket
import time

HOST = "10.136.49.55" # Tablet on eduroam - The server's hostname or IP address
PORT = 4000 # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
	s.connect((HOST, PORT))
	while True:
		s.sendall(b"Testing TCP Connection")
		print('sent data')
		data = s.recv(1024)
		print(data)
		#if data != "Hello Client!":
		#	raise RuntimeError("Connection Is Not Secured")
		time.sleep(10)
