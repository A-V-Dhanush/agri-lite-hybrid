#!/bin/bash
# =============================================================================
# AgriLite-Hybrid - Raspberry Pi Setup Script
# setup_pi.sh - Installation and configuration
# =============================================================================

set -e

echo "=============================================="
echo "AgriLite-Hybrid Raspberry Pi Setup"
echo "=============================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Device: $(cat /proc/device-tree/model)"
fi

# Update system packages
echo ""
echo "[1/6] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    libpng-dev \
    libcamera-dev \
    libcamera-apps

# Enable camera if not already enabled
echo ""
echo "[3/6] Checking camera configuration..."
if ! grep -q "^start_x=1" /boot/config.txt 2>/dev/null; then
    echo "Note: If using legacy camera, you may need to enable it in raspi-config"
fi

# Create virtual environment
echo ""
echo "[4/6] Creating Python virtual environment..."
VENV_DIR="$HOME/agrilite_venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip3 install --upgrade pip

# Install Python packages
echo ""
echo "[5/6] Installing Python packages..."
pip3 install -r requirements_pi.txt

# Create models directory
echo ""
echo "[6/6] Setting up directories..."
mkdir -p models
mkdir -p captures

# Create symbolic link for class labels if shared exists
if [ -f "../../models/class_labels.json" ]; then
    ln -sf "../../models/class_labels.json" models/class_labels.json
    echo "Linked class_labels.json from shared models directory"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy your TFLite model to: $(pwd)/models/agrilite_model.tflite"
echo "2. Activate virtual environment: source $VENV_DIR/bin/activate"
echo "3. Run the web server: python3 web_server_pi.py"
echo ""
echo "Optional: Add to startup"
echo "  sudo nano /etc/rc.local"
echo "  Add before 'exit 0':"
echo "    cd $(pwd) && source $VENV_DIR/bin/activate && python3 web_server_pi.py &"
echo ""
