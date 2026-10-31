#!/home/noah/coin-sorter/env/bin/python3

import os
import time
import datetime
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
BRIGHTNESS = 0.6

IMAGES_PER_COLOR = 10
ROTATION_DELAY = 0.01

WHITE_FILL = 0.3
BASE_FACTOR = 0.45
PEAK_FACTOR = 1.00
SIDE_FACTOR = 0.8
CLUSTER_SIZE = 1

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255)
]

BASE_DIR = "/home/noah/coin-sorter/data/raw"

# -----------------------------
# SETUP
# -----------------------------

pixels = neopixel.NeoPixel(
    LED_PIN,
    NUM_LEDS,
    brightness=BRIGHTNESS,
    auto_write=True
)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = os.path.join(BASE_DIR, f"coin_{timestamp}")
os.makedirs(session_dir, exist_ok=True)

print(f"Saving images to: {session_dir}")

# -----------------------------
# PRECOMPUTE ROTATION PATTERNS
# -----------------------------

def precompute_patterns(color):
    patterns = []
    for center in range(NUM_LEDS):
        frame = []
        for led in range(NUM_LEDS):
            d = min(abs(led - center), NUM_LEDS - abs(led - center))

            if d == 0:
                factor = PEAK_FACTOR
            elif d <= CLUSTER_SIZE:
                factor = SIDE_FACTOR
            else:
                factor = BASE_FACTOR

            r = min(255, int(color[0] * factor + 255 * WHITE_FILL))
            g = min(255, int(color[1] * factor + 255 * WHITE_FILL))
            b = min(255, int(color[2] * factor + 255 * WHITE_FILL))

            frame.append((r, g, b))

        patterns.append(frame)
    return patterns

# -----------------------------
# MAIN LOOP
# -----------------------------

try:
    img_index = 0

    for color in COLORS:
        print(f"\n=== COLOR {color} ===")

        patterns = precompute_patterns(color)

        for i in range(IMAGES_PER_COLOR):

            pixels[:] = patterns[i % NUM_LEDS]
            time.sleep(ROTATION_DELAY)

            filename = os.path.join(session_dir, f"img_{img_index:04d}.jpg")

            # FAST USB CAMERA CAPTURE
            subprocess.run([
                "ffmpeg",
                "-f", "v4l2",
                "-input_format", "mjpeg",
                "-video_size", f"{RES_X}x{RES_Y}",
                "-i", CAMERA_DEVICE,
                "-vframes", "1",
                filename,
                "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print(f"Captured {filename}")
            img_index += 1

    print("Capture complete!")

except KeyboardInterrupt:
    print("\nCtrl+C detected — shutting down LEDs and exiting safely.")

finally:
    pixels.fill((0, 0, 0))
    print("LEDs off. Goodbye.")


