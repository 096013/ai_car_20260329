# -*- coding: utf-8 -*-
import math
import threading
import time

try:
    from gpiozero import OutputDevice
except ImportError:  # pragma: no cover - fallback for non-hardware environments
    class OutputDevice:
        def __init__(self, pin):
            self.pin = pin
            self.value = 0

        def on(self):
            self.value = 1

        def off(self):
            self.value = 0

try:
    import smbus
except ImportError:  # pragma: no cover - fallback for non-hardware environments
    smbus = None


class DummySMBus:
    def __init__(self, bus_id):
        self.bus_id = bus_id
        self.memory = {}

    def write_byte_data(self, addr, reg, val):
        self.memory[(addr, reg)] = val

    def read_byte_data(self, addr, reg):
        return self.memory.get((addr, reg), 0)


_i2c = smbus.SMBus(1) if smbus is not None else DummySMBus(1)


class PCA9685:
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06

    def __init__(self, addr=0x40):
        self.addr = addr
        self.write(self.__MODE1, 0x00)

    def write(self, reg, val):
        _i2c.write_byte_data(self.addr, reg, val)

    def read(self, reg):
        return _i2c.read_byte_data(self.addr, reg)

    def setPWMFreq(self, freq):
        prescaleval = 25000000.0 / 4096.0 / float(freq) - 1.0
        prescale = int(math.floor(prescaleval + 0.5))
        old = self.read(self.__MODE1)
        self.write(self.__MODE1, (old & 0x7F) | 0x10)
        self.write(self.__PRESCALE, prescale)
        self.write(self.__MODE1, old)
        time.sleep(0.005)
        self.write(self.__MODE1, old | 0x80)

    def setPWM(self, ch, on, off):
        base = self.__LED0_ON_L + 4 * ch
        self.write(base, on & 0xFF)
        self.write(base + 1, on >> 8)
        self.write(base + 2, off & 0xFF)
        self.write(base + 3, off >> 8)

    def setDutycycle(self, ch, duty):
        duty = max(0, min(100, int(duty)))
        off = int(duty * 4095 / 100.0)
        self.setPWM(ch, 0, off)

    def setLevel(self, ch, value):
        self.setPWM(ch, 0, 4095 if value else 0)


class RobotControl:
    def __init__(self):
        self.lock = threading.RLock()
        self.pwm = PCA9685(0x40)
        self.pwm.setPWMFreq(50)

        self.motorD1 = OutputDevice(25)
        self.motorD2 = OutputDevice(24)
        self.current_speed = 70
        self.stop()

    def move(self, action, speed=None):
        with self.lock:
            if speed is not None:
                self.current_speed = max(0, min(100, int(speed)))

            if action == "forward":
                self.drive(self.current_speed, 0)
            elif action == "backward":
                self.drive(-self.current_speed, 0)
            elif action == "left":
                self.drive(0, -self.current_speed)
            elif action == "right":
                self.drive(0, self.current_speed)
            else:
                self.stop()

    def drive(self, forward_speed, turn_rate=0):
        with self.lock:
            left_speed = max(-100, min(100, int(forward_speed - turn_rate)))
            right_speed = max(-100, min(100, int(forward_speed + turn_rate)))

            self._set_motor("A", left_speed)
            self._set_motor("C", left_speed)
            self._set_motor("B", right_speed)
            self._set_motor("D", right_speed)

    def stop(self):
        with self.lock:
            for channel in [0, 5, 6, 11]:
                self.pwm.setDutycycle(channel, 0)
            self.motorD1.off()
            self.motorD2.off()

    def _set_motor(self, name, signed_speed):
        speed = abs(int(signed_speed))
        direction = "forward" if signed_speed >= 0 else "backward"

        if name == "A":
            self.pwm.setDutycycle(0, speed)
            self.pwm.setLevel(2, 0 if direction == "forward" else 1)
            self.pwm.setLevel(1, 1 if direction == "forward" else 0)
        elif name == "B":
            self.pwm.setDutycycle(5, speed)
            self.pwm.setLevel(3, 1 if direction == "forward" else 0)
            self.pwm.setLevel(4, 0 if direction == "forward" else 1)
        elif name == "C":
            self.pwm.setDutycycle(6, speed)
            self.pwm.setLevel(8, 1 if direction == "forward" else 0)
            self.pwm.setLevel(7, 0 if direction == "forward" else 1)
        elif name == "D":
            self.pwm.setDutycycle(11, speed)
            if direction == "forward":
                self.motorD1.off()
                self.motorD2.on()
            else:
                self.motorD1.on()
                self.motorD2.off()
