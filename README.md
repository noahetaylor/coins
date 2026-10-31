# Coin Date Reader

By **Noah Taylor** · March 2025 – present

Reads the mint year on worn US quarters. A Raspberry Pi camera rig photographs
each coin under directional (raking) LED light, and a ResNet-18 date reader
crops the date band and reads its four digits.

- 99.1% of coins read correctly on 5-fold cross-validation over 751 coins
- Raking-light capture raised the rig coins from 42 to 44 of 50

| Folder | Contents |
|---|---|
| [`coins/important/`](coins/important/) | Date-band extraction, the date reader, training and evaluation ([README](coins/important/README.md)) |
| [`coins/rig/`](coins/rig/) | Raspberry Pi capture scripts, LED control and the one-button scan page |

The earliest files in `coins/` (e.g. `lab1.py`, `exercise2.py`) are exercises
from the TensorFlow course I learned with; the current pipeline uses PyTorch.
