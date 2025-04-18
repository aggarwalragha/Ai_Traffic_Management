# SmartAiTrafficManagement

This project is a **Smart AI-based Traffic Management System** that combines **computer vision (YOLOv2)** with **real-time traffic simulation (Pygame)** to detect vehicles from images and simulate dynamic signal control.

##  Features

-  Uses **YOLOv2 object detection** to identify vehicles (`car`, `bus`, `truck`, `motorbike`) from traffic images.
-  Simulates a 4-way traffic intersection using **Pygame**.
-  Automatically adjusts signal timings based on the number of detected vehicles.
-  Supports image-based input for vehicle analysis (future support for video/real-time feeds possible).

---

##  Tech Stack

| Layer        | Technology             |
|-------------|------------------------|
| ML Backend  | YOLOv2 (OpenCV DNN)    |
| Simulation  | Python with Pygame     |
| Language    | Python 3.11+           |
| Detection   | COCO-trained YOLOv2    |



