# ⚽ Yolo Soccer: Real-Time Player Tracking and Visualization with YOLOv8

Welcome to my senior research project! This Python-based project was the culmination of months of interdisciplinary work combining computer vision, machine learning, and my deep love for sports and technology. I developed it during my senior year at Thomas Jefferson High School for Science and Technology (TJHSST) as part of my capstone experience.

## 📖 About the Project

**Yolo Soccer** uses the YOLOv8 object detection model to:
- Detect soccer players in real-time footage.
- Track player positions on a bird’s-eye view of the soccer field.
- Classify players into teams based on jersey color.
- Generate heatmaps showing player density over time.

This project has potential applications in:
- Automated sports analytics.
- Player tracking for training and strategy.
- Enriching viewer experience in broadcasts.

## 💡 My Journey

As a CS student passionate about machine learning and computer vision, I wanted a project that pushed the limits of what I could do with deep learning frameworks. I started by experimenting with basic YOLOv8 models and gradually incorporated OpenCV, Matplotlib, and other libraries to create a full pipeline from detection to visualization.

I took inspiration from broadcast-level sports tracking systems and wanted to build a low-cost, accessible alternative that students and hobbyists could use. This project was deeply meaningful to me—it sharpened my skills, gave me hands-on experience with real-world challenges, and confirmed my commitment to studying machine learning and AI at the University of Maryland.

## 🚀 How It Works

The script does the following:
1. Loads a video of a soccer game and initializes the YOLOv8 model.
2. Asks the user to:
   - Select the corners of the field for perspective warping.
   - Identify field color bounds.
   - Select jersey colors of both teams.
3. Uses YOLOv8 to detect players in overlapping patches.
4. Transforms player positions into a bird’s-eye view using a perspective matrix.
5. Classifies players by jersey color using simple color-distance metrics.
6. Displays:
   - A live frame with bounding boxes.
   - A live bird’s-eye field with player positions.
   - A real-time heatmap of player positions.
7. Outputs a side-by-side video and saves a heatmap.

## 📦 Dependencies

Install required packages with:

```bash
pip install opencv-python-headless ultralytics matplotlib pillow scipy keyboard numpy
```

## 📂 Input

- A sample video file path should be specified in the `filename` variable (`.MP4` format). (Ex. files here: https://drive.google.com/drive/folders/1v4c_Z-YZUsfC4u7KFTjWYzjPg3CZwkyr?usp=sharing)
- A reference image of a soccer field (`Soccer_field.png`) for the bird’s-eye overlay.

## 🛠️ Usage

Run the script:

```bash
python "Yolo soccer thingy.py"
```

### Controls:
- Use mouse clicks to define field and jersey regions.
- Press `Enter` after selecting the green field area.
- Press `S` to save the current video frame.
- Press `Q` to quit and save the video and heatmap.

The program will produce:
- `output.mp4`: A side-by-side video showing the original footage and bird’s-eye tracking.
- `heatmap.png`: A colorized heatmap image of player movement.

## 🧠 Technical Highlights

- **YOLOv8 (via Ultralytics)**: For real-time player detection.
- **OpenCV**: For image manipulation, warping, and GUI interactions.
- **Matplotlib 3D Scatter Plotting**: For live jersey color analysis.
- **Heatmap Aggregation**: Based on field-warped player positions.

## 🎓 About Me

I'm Lucas Tao, currently a CS major at the University of Maryland. I'm deeply interested in machine learning, computer vision, and building systems that connect the digital and physical worlds. This project reflects both my technical skills and my curiosity-driven approach to solving real-world problems.
