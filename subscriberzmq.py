# import zmq

# port =5002
# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# # We can connect to several endpoints if we desire, and receive from all.
# socket.connect("tcp://192.168.18.178:%s" % port)

# # We must declare the socket as of type SUBSCRIBER, and pass a prefix filter.
# # Here, the filter is the empty string, wich means we receive all messages.
# # We may subscribe to several filters, thus receiving from all.
# socket.setsockopt(zmq.SUBSCRIBE, b'')

# message = socket.recv_pyobj()
# print (message.get(1)[2])

# import sys
# import zmq

# port = "5556"
# if len(sys.argv) > 1:
#     port =  sys.argv[1]
#     int(port)
    
# if len(sys.argv) > 2:
#     port1 =  sys.argv[2]
#     int(port1)

# # Socket to talk to server
# context = zmq.Context()
# socket = context.socket(zmq.SUB)

# print ("Collecting updates from weather server...")
# socket.connect ("tcp://192.168.18.178:%s" % port)


# # Subscribe to zipcode, default is NYC, 10001
# topicfilter = b"10001"
# socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

# # Process 5 updates
# total_value = 0
# for update_nbr in range (5):
#     string = socket.recv()
#     topic, messagedata = string.split()
#     total_value += int(messagedata)
#     print (topic, messagedata)

# print ("Average messagedata value for topic '%s' was %dF" % (topicfilter, total_value / update_nbr))
      
# import zmq

# context = zmq.Context()

# #  Socket to talk to server
# print("Connecting to hello world server...")
# socket = context.socket(zmq.REQ)
# # socket.connect("tcp://localhost:5555")
# socket.connect("tcp://192.168.18.178:5555")


# #  Do 10 requests, waiting each time for a response
# for request in range(10):
#     print("Sending request %s ..." % request)
#     socket.send(b"Hello")

#     #  Get the reply.
#     message = socket.recv()
#     print("Received reply %s [ %s ]" % (request, message))

import sys
import time
import zmq

context = zmq.Context()

# Socket to receive messages on
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5558")
# receiver.bind("tcp://*92.168.18.178:5558")


while True:
	data = receiver.recv()
	print(data)