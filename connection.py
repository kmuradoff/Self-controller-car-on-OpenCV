import time
import serial

if __name__ == '__main__':
   ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
   ser.reset_input_buffer()

   while True:
       if ser.in_waiting > 0:
           ser.write(b"forward\n")
           time.sleep(1)
           ser.write(b"stop\n")
           time.sleep(1)
           ser.write(b"backward\n")
           time.sleep(1)