import sys
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# =========================================================================
# 1. PATH OVERRIDE (Failsafe for the college PC)
# =========================================================================
PAL_PATH = r"C:\Users\Admin\Desktop\Q-car\Quanser_Academic_Resources-dev-windows\Quanser_Academic_Resources-dev-windows\0_libraries\python"
if PAL_PATH not in sys.path:
    sys.path.append(PAL_PATH)

try:
    from pal.products.qcar import QCarCameras
except ImportError as e:
    print(f"❌ Error importing PAL: {e}")
    sys.exit()

# =========================================================================
# 2. YOLO VISION LOOP
# =========================================================================
def main():
    print("🧠 Loading YOLOv8 Brain (best.pt)...")
    model = YOLO('best.pt')

    print("📷 Booting Front Camera...")
    # Enable ONLY the front camera to ensure smooth 30 FPS
    cameras = QCarCameras(enableBack=False, enableFront=True, enableLeft=False, enableRight=False)

    print("🟢 Vision Agent Active! Press 'q' in the video window to quit.")

    # The 'with' statement safely handles starting and stopping the hardware
    with cameras:
        try:
            while True:
                cameras.readAll()
                
                # Extract the image array from the active camera
                front_img = None
                for c in cameras.csi:
                    if c is not None and c.imageData is not None:
                        front_img = c.imageData
                        break
                
                if front_img is not None:
                    # Run YOLO Inference
                    results = model.predict(source=front_img, conf=0.40, verbose=False)
                    
                    # Draw the bounding boxes
                    annotated_frame = results[0].plot()

                    # Display the feed
                    cv2.imshow("Black Hawks Vision", annotated_frame)

                # Safety break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Manual override detected.")

    cv2.destroyAllWindows()
    print("Safely shut down.")

if __name__ == '__main__':
    main()