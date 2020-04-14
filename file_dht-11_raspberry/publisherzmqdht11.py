import zmq
import random
import time
import sys
import Adafruit_DHT

context = zmq.Context()

# Socket with direct access to the sink: used to syncronize start of batch
sink = context.socket(zmq.PUSH)
# sink.connect("tcp://localhost:5558")
sink.connect("tcp://192.168.18.178:5558")

while True:
	humidity, temperature = Adafruit_DHT.read_retry(11, 4)
	data = str('Temp: {0:0.1f} C  Humidity: {1:0.1f} %'.format(temperature, humidity))
	print ('Temp: {0:0.1f} C  Humidity: {1:0.1f} %'.format(temperature, humidity))
	# data = []
	# data.append(humidity)
	# data.append(temperature)
	# sink.send(b'workload')
	# sink.send(str([humidity,temperature]).encode('ascii'))
	sink.send(data.encode('ascii'))

