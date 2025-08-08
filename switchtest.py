import serial
import serial.tools.list_ports
import time


def find_arduino_port():
    """
    finds a serial port that connects to an Arduino
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description or "ttyACM" in port.device or "usbmodem" in port.device:
            return port.device
        return None


ARDUINO_PORT = find_arduino_port()
KEYSET = {"a": 0, "w": 1, "e": 2, "r": 3}
with serial.Serial(ARDUINO_PORT, 9600, timeout=0) as ser:
    print('Arduino booting. Please wait\n')
    time.sleep(2)  # gives the arduino time to reset
    input('Arduino ready. Press Enter to start\n')
    ser.write(bytes([0]))

    while True:
        a = input()
        ser.write(bytes([KEYSET[a]]))
