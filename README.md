# 🏓 Rainbow Pong RL

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

**Rainbow DQN agent** mastering Pong against a physics-based opponent

## ✨ Key Features
- 🚀 **Headless training** (`--headless` flag for 20x faster training)
- 🌈 **Full Rainbow DQN** (Noisy Nets + PER + Distributional RL)
- 🤖 **Smart physics opponent** (With deliberate limitations)
- 📊 **Live training visualizer** (Metrics saved to `logs/`)

## 🚦 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train with visualization (Recommended for debugging)
python main.py

# Train in headless mode (For serious training)
python main.py --headless --episodes 10000

# Watch trained AI play
python main.py --load models/trained.pth