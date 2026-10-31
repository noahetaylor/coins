import board, neopixel, time
pixels = neopixel.NeoPixel(board.D18, 24, brightness=0.6, auto_write=True)

WHITE_FILL = 0.20
BASE_FACTOR = 0.05
PEAK_FACTOR = 1.00
SIDE_FACTOR = 0.40
CLUSTER_SIZE = 1

while True:
    for center in range(24):
        for led in range(24):
            d = min(abs(led - center), 24 - abs(led - center))

            if d == 0:
                factor = PEAK_FACTOR
            elif d <= CLUSTER_SIZE:
                factor = SIDE_FACTOR
            else:
                factor = BASE_FACTOR

            r = int(255 * factor + 255 * WHITE_FILL)
            g = int(0   * factor + 255 * WHITE_FILL)
            b = int(0   * factor + 255 * WHITE_FILL)

            r = min(255, r)
            g = min(255, g)
            b = min(255, b)

            pixels[led] = (r, g, b)

        time.sleep(0.1)
