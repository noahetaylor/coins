import os
import time
import subprocess
import board
import neopixel

# -----------------------------
# CONFIG
# -----------------------------

CAMERA_DEVICE = "/dev/video0"
RES_X = 4656
RES_Y = 3496

LED_PIN = board.D18
NUM_LEDS = 24
BRIGHTNESS = 0.8

COLOR = (255, 255, 255)  # white

OUTPUT = "/home/noah/coin-sorter/output/single_lit.jpg"

# -----------------------------
# SETUP
# -----------------------------

pixels = neopixel.NeoPixel(
    LED_PIN,
    NUM_LEDS,
    brightness=BRIGHTNESS,
    auto_write=True
)

# -----------------------------
# CAPTURE
# -----------------------------

pixels.fill(COLOR)
time.sleep(0.2)

cmd = [
    "fswebcam",
    "--no-banner",
    "-r", f"{RES_X}x{RES_Y}",
    "-d", CAMERA_DEVICE,
    OUTPUT
]

print(f"Capturing {OUTPUT}")
subprocess.run(cmd)

pixels.fill((0, 0, 0))
print("Done.")
