This repository provides setup instructions, dependencies installation, and usage guide for running the simulation package with ROS 2.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)

   * [System Dependencies](#system-dependencies)
   * [Python Packages](#python-packages)
   * [Open Motion Planning Library (OMPL)](#open-motion-planning-library-ompl)
   * [ACADO Toolkit](#acado-toolkit)
3. [Building the Package](#building-the-package)
4. [Usage Guide](#usage-guide)

---

## Prerequisites

* Ubuntu 22 or later
* ROS 2 (Foxy, Galactic, or Humble) installed and sourced
* `cmake`, `git`, and `build-essential` packages

---

## Installation

### System Dependencies

Update package lists and upgrade existing packages:

```bash
sudo apt update && sudo apt upgrade -y
```

Install development libraries:

```bash
sudo apt install -y \
    libopencv-dev \
    libboost-all-dev \
    libompl-dev ompl-demos \
    libeigen3-dev
```

### Python Packages

Install required Python modules:

```bash
pip install numpy matplotlib scikit-learn opencv-python scipy pygame pyserial
```

### Open Motion Planning Library (OMPL)

Download and run the OMPL installation script:

```bash
wget https://ompl.kavrakilab.org/install-ompl-ubuntu.sh
chmod u+x install-ompl-ubuntu.sh
./install-ompl-ubuntu.sh
```

### ACADO Toolkit

Clone, build, and install ACADO:

```bash
git clone https://github.com/acado/acado.git
cd acado
mkdir build && cd build
cmake ..
make
sudo make install

# Ensure the library path is updated
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

---

## Building the Package

In your ROS 2 workspace (e.g., `~/decision_module`), build :

```bash
cd ~/DecisionModule
colcon build
source install/setup.bash
```

---

## Usage Guide

### Launching the Simulation

Launch the full simulation environment:

```bash
ros2 launch simulation_pkg decision.launch.py
```

### Running the Decision Node

Run only the decision node:

```bash
ros2 run simulation_pkg decision
```

---

## Joystick Controller

The joystick controller allows you to manually steer opponent bots during simulation. Setup is provided by an external repository:

* Joystick setup instructions and configuration files: [Joystick Setup Repository](https://github.com/experiment322/controlloid-server)

1. **Install and configure** the joystick package following the steps in the linked repository.
2. **Run the joystick controller node**:

   ```bash
   ros2 run simulation_pkg joystick_controller
   ```
3. **Run the server**:
   ```bash
   ./start.sh 
   ```

4. **Connect multiple joystick apps** to control multiple bots simultaneously—each instance will drive a separate bot.
5. **Controller layout**:

   * Refer to the screenshot below and configure your controller as shown.
   * The upper button on the joystick can be used to switch control to the currently active bot.
     
![445f77af-5178-48e8-9bdf-bc324b348c51](https://github.com/user-attachments/assets/fc81f2ab-c6af-41f7-ae1d-a5483c5070bf)



---

## License & Acknowledgments

* Original dependencies and scripts provided by the OMPL and ACADO teams.

