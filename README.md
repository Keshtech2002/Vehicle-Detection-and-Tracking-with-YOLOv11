# 🚗 Vehicle & Object Detection and Counting App

A Streamlit web app that detects and counts objects (vehicles, people, etc.) in **images** or **videos** using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).  
The app displays annotated images/videos with bounding boxes and shows live counts in a nice dashboard interface.

---

## ✨ Features

- 🖼️ **Image mode** – upload a single image to get annotated output + class counts  
- 🎥 **Video mode** – upload a video; the app runs YOLO tracking, draws a horizontal counting line, and counts each object when it crosses the line  
- ⏱️ **Live updates** – shows per-frame preview, running counts, and progress bar while processing long videos  
- 📊 **Sorted count table** – displays detected classes sorted by frequency  
- 💾 **Download** annotated images/videos directly from the interface  

---

## 📂 Project Structure

Click the links to view each file/folder on GitHub:

| FILES | PURPOSE |
| :--- | :----- |
| [app.py](./app.py) | Streamlite app for UI interraction by the user|
| [image_counting.py](./image_counting.py) | py file to count vehicles contained in images |
| [video_counting.py](./video_counting.py) | py file to count vehicles contained in a video|
| [vehicle_detection.ipynb](./vehicle_detection.ipynb) | Notebook initially created for experiment |
| [annotated_assets](./annotated_assets/) | Directory of results obtained from testing the application |
| [test_assets](./test_assets/) | Test images and videos |
| [yolo11l.pt](./yolo11l.pt) | Model Used |
| [README.md](./README.md) | Readme file |
| [requirements.txt](./requirements.txt) | Dependencies |

## 🤖 Model Used

This app uses **YOLOv8** from [Ultralytics](https://github.com/ultralytics/ultralytics).  
- Pretrained weights: `yolo11l.pt`  
- Supports object detection with tracking IDs  
- You can replace the `.pt` file with any YOLOv8 weights you like.

---

## ⚙️ Installation & Setup

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # on Linux/Mac
venv\Scripts\activate     # on Windows
```

### 3. Install dependencies

Create a requirements.txt containing:
```bash
streamlit
ultralytics
opencv-python
pillow
pandas
numpy
```

Then run:
```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app
```
Open the local URL that Streamlit prints (e.g. ```http://localhost:8501```).

## 📸 Usage

- Image mode: Choose “Image” from the sidebar, upload a picture, see annotated output + counts.

- Video mode: Choose “Video,” upload a video, set the line position and update interval, click “Run detection & tracking.”

- Download annotated outputs when processing finishes.

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the project

2. Create your feature branch ```git checkout -b feature/AmazingFeature```

3. Commit your changes ```git commit -m 'Add some AmazingFeature'```

4. Push to the branch ```git push origin feature/AmazingFeature```

5. Open a Pull Request


## 📬 Contact

Created by [Keshinro Mus'ab] – feel free to contact me:

Email: [keshtech2002@gmail.com](keshtech2002@gmail.com)

LinkedIn: [Keshinro Mus'ab](https://www.linkedin.com/in/keshinro-musab/)

Twitter: [Keshinro Mus'ab](https://x.com/MusKhayr)


