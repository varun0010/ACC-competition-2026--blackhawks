# ACC-competition-2026--blackhawks
This repository contains codes for Quanser 2026 ACC Self Driving Car Competition - Blackhawks

## Instruction to run the code

1) Make sure the pretrained YOLOv8 models (`best.pt` and `yolov8n.pt`) are downloaded and kept in the " code " folder  
*(These are already included in this repository).*
    
2) To install all the required libraries:
```
python install.py
```
*(Alternatively, you can manually run `pip install -r requirements.txt`)*

3) To initialize the environment and launch the agent on the QCar:
```
IGNITE_CAR.bat
```
*(This batch file handles the environment setup and runs `Python code/acc_master_agent.py`)*

4) To test the different scenarios individually:
```
python code/scenario.py
```

5) To run the data collection module independently:
```
python code/acc_data_collector.py
```

## Track Details
*   `code/camera_waypoints.csv` contains the navigation waypoints.
