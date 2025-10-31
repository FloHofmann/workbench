"""
This script will find an arduino port and start generating keypresses
to trigger the click sounds in spike2 as well as control the
multiplexer output by sending the respective bytes through the
arduino's port
"""
import serial
import serial.tools.list_ports
import time
import random
import keyboard


def find_arduino_port():
    """
    finds a serial port that connects to an Arduino
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description or "ttyACM" in port.device or "usbmodem" in port.device:
            return port.device
        return None


if __name__ == '__main__':
    REPS = 30
    ISI = 2.5  # seconds
    KEYSET = {"a": 0, "w": 1, "e": 2, "r": 3}
    keyset_entries = KEYSET.keys()
    keylist = sum([random.sample(list(keyset_entries), 4)
                  for _ in range(REPS)], [])
    print(keylist)
    ARDUINO_PORT = find_arduino_port()
    if ARDUINO_PORT is None:
        print("No Arduino found\n quitting the script")
        exit()

    # Connect to the Arduino
    with serial.Serial(ARDUINO_PORT, 9600, timeout=0) as ser:
        print('Arduino booting. Please wait\n')
        time.sleep(2)  # gives the arduino time to reset
        input('Arduino ready. Press Enter to start\n')

        # here follows the logic to send the bytes
        for idx, key in enumerate(keylist):
            # writes the number associated with the key as byte through the serial port
            print("Sending key {} to Arduino".format(KEYSET[key]))
            ser.write(bytes([KEYSET[key]]))
            time.sleep(ISI+random.randint(30, 90)/60)
            keyboard.press(key)
            time.sleep(random.uniform(0.1, 0.5))
            keyboard.release(key)

            print("{:.1f}% === {}".format(idx/len(keylist)*100, key))

        print('Done... Closing serial port')
