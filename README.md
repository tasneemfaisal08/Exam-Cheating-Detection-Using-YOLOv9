#  Cheating Exam Detection System

A real-time exam cheating detection system using **YOLOv3** object detection to identify forbidden items (cell phones, books) during exams — via webcam or video file.

---


##  Overview

This project uses the YOLOv3 deep learning model to detect cheating behavior in exam environments. It processes video frames in real-time and flags any detected **cell phone** or **book**, drawing bounding boxes around them and saving the annotated output video.

---

##  Features

- Real-time object detection using **YOLOv3**
- Detects cheating-related objects: `cell phone`, `book`
- Draws **red bounding boxes** around detected items
- Saves the output as an annotated `.avi` video file
- Optional **Streamlit web app** interface with live webcam support via `streamlit-webrtc`

---

## 📁 Project Structure

```
project/
│
├── objectDetection.py          # Core detection script (webcam/video)
├── real_streamlit_api.py       # Streamlit web app interface
├── cheating_exam_detection.ipynb  # Jupyter Notebook version
│
├── yolov3.weights              # Pre-trained YOLOv3 weights
├── yolov3-416.cfg              # YOLOv3 model configuration
├── coco.names                  # COCO class labels (80 classes)
│
├── 1230.mp4                    # Sample input video
├── detected_video.avi          # Output: detection results
└── output_video.avi            # Output: processed video
```

---

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Streamlit *(for web interface only)*
- streamlit-webrtc *(for live webcam in web interface)*
- PyTorch *(for Streamlit version)*

---

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/cheating-exam-detection.git
   cd cheating-exam-detection
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python numpy streamlit streamlit-webrtc torch
   ```

3. **Download YOLOv3 weights** *(if not included)*
   ```bash
   wget https://pjreddie.com/media/files/yolov3.weights
   ```

---

## Usage

### Option 1: Run the core detection script

```bash
python objectDetection.py
```

- Opens your **webcam** by default (`cv2.VideoCapture(0)`)
- Press **`q`** to quit
- Output saved to `detected_video.avi`

> To use a video file instead of webcam, change:
> ```python
> cap = cv2.VideoCapture(0)
> # to:
> cap = cv2.VideoCapture('your_video.mp4')
> ```

### Option 2: Run the Streamlit web app

```bash
streamlit run real_streamlit_api.py
```

- Opens a browser-based UI
- Supports live webcam streaming and side-by-side video comparison

---

##  How It Works

1. **Model Loading** — YOLOv3 is loaded using OpenCV's `dnn` module with the provided `.weights` and `.cfg` files.
2. **Frame Processing** — Each video frame is converted to a blob and passed through the network.
3. **Detection** — The model outputs bounding boxes and confidence scores for all 80 COCO classes.
4. **Filtering** — Only `cell phone` and `book` detections above a confidence threshold of `0.5` are kept (using NMS with threshold `0.4`).
5. **Annotation** — Detected objects are highlighted with **red rectangles** and labeled with the class name and confidence score.
6. **Output** — The annotated frames are written to `detected_video.avi`.

---

##  Output

| Item | Description |
|------|-------------|
| `detected_video.avi` | Annotated video with bounding boxes |
| Red bounding box | Indicates a detected cheating item |
| Blue label text | Shows class name and confidence score |

---

##  Notes

- The model uses **608×608** input resolution as defined in `yolov3-416.cfg` for high accuracy.
- Detection blob size is set to **640×640** in the script — adjust for speed vs. accuracy trade-off.
- The confidence threshold is set to `0.1` for detection and `0.5` for NMS filtering.
