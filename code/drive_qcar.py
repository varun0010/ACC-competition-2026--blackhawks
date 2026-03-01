import sys
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# =========================================================================
# 1. THE QVL LINK (No QUARC License Required!)
# =========================================================================
FALLBACK_PATH = r"C:\Users\Admin\Desktop\Q-car\Quanser_Academic_Resources-dev-windows\Quanser_Academic_Resources-dev-windows"
QAL_DIR = os.environ.get('QAL_DIR', FALLBACK_PATH)

# Add the Quanser Python API to the system path
sys.path.append(os.path.join(QAL_DIR, '0_libraries', 'python'))

try:
    # Importing pure QVL modules
    from qvl.qlabs import QuanserInteractiveLabs
    from qvl.qcar2 import QLabsQCar2
    # We will use the free camera or QCar cameras for vision next
    from qvl.free_camera import QLabsFreeCamera 
except ImportError as e:
    print(f"❌ Error importing QVL modules: {e}")
    sys.exit()

# =========================================================================
# 2. MAIN DRIVING LOOP
# =========================================================================
def main():
    print("🧠 Loading YOLOv8 Brain (best.pt)...")
    # model = YOLO('best.pt') # Commented out just for this initial connection test

    print("🔌 Connecting to QLabs Simulator...")
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("❌ Failed to connect to QLabs. Make sure the simulator is running.")
        sys.exit()

    print("🚗 Linking to the QCar2...")
    myCar = QLabsQCar2(qlabs)
    
    # We connect to Actor ID 0 (the car spawned by your MATLAB script)
    myCar.actorNumber = 0 
    
    print("🟢 Agent Active! Sending test movement command...")
    
    try:
        # 1. Drive forward (0.5 m/s, 0.0 steering, all lights False)
        myCar.set_velocity_and_request_state(0.5, 0.0, False, False, False, False, False)
        
        # Let it drive for 3 seconds
        time.sleep(3)
        
        # 2. Stop the car (0.0 m/s, 0.0 steering, all lights False)
        print("🛑 Stopping car.")
        myCar.set_velocity_and_request_state(0.0, 0.0, False, False, False, False, False)

    except KeyboardInterrupt:
        print("\n🛑 Manual override detected.")
        
    finally:
        # 3. Safe Cleanup (0.0 m/s, 0.0 steering, all lights False)
        myCar.set_velocity_and_request_state(0.0, 0.0, False, False, False, False, False)
        qlabs.close()
        print("Disconnected successfully.")

if __name__ == '__main__':
    main()