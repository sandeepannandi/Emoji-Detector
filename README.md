# Emoji Reactor

A real-time camera-based emoji display application that uses MediaPipe to detect your poses and facial expressions, then displays corresponding emojis in a separate window.

## Features

- **Hand Detection**: Raises hands above shoulders → displays hands up emoji 🙌
- **Smile Detection**: Detects smiling → displays smiling emoji 😊  
- **Default State**: Straight face → displays neutral emoji 😐
- **Real-time Processing**: Live camera feed with instant emoji reactions

## Requirements

- Python 3.12 (Homebrew: `brew install python@3.12`)
- macOS or Windows with a webcam
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone or download this project**

2. **Create a virtual environment (Python 3.12) and install deps:**
   ```bash
   # macOS: ensure Python 3.12 is installed
   brew install python@3.12

   # Create and activate a virtual environment
   python3.12 -m venv emoji_env
   source emoji_env/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Ensure you have the emoji images in the project directory:**
   - `smile.jpg` - Smiling face emoji
   - `plain.png` - Straight face emoji  
   - `air.jpg` - Hands up emoji

## Usage

1. **Run the application:**
   ```bash
   # Option A: use helper script
   ./run.sh

   # Option B: run manually
   source emoji_env/bin/activate
   python emoji_reactor.py
   ```

2. **Two windows will open:**
   - **Camera Feed**: Shows your live camera with detection status
   - **Emoji Output**: Displays the corresponding emoji based on your actions

3. **Controls:**
   - Press `q` to quit the application
   - Raise your hands above your shoulders for hands up emoji
   - Smile for the smiling emoji
   - Keep a straight face for the neutral emoji

## How It Works

The application uses two MediaPipe solutions:

1. **Pose Detection**: Monitors shoulder and wrist positions to detect raised hands
2. **Face Mesh Detection**: Analyzes mouth shape to detect smiling vs. straight face

### Detection Priority
1. **Hands Up** (highest priority) - Overrides facial expression detection
2. **Smiling** - Detected when mouth aspect ratio exceeds threshold
3. **Straight Face** - Default state when no smile is detected

## Customization

### Adjusting Smile Sensitivity
Edit the `SMILE_THRESHOLD` value in `emoji_reactor.py`:
- Decrease value (e.g., 0.30) if smiles aren't detected
- Increase value (e.g., 0.40) if false positive smiles occur

### Changing Emoji Images
Replace the image files with your own:
- `smile.jpg` - Your smiling emoji
- `plain.png` - Your neutral emoji
- `air.jpg` - Your hands up emoji

## Troubleshooting

### Camera Issues (macOS)
- If you see "not authorized to capture video", grant Camera access for your terminal/editor:
  - System Settings → Privacy & Security → Camera → enable for Terminal/VS Code/iTerm
- Quit and relaunch the terminal/editor after changing permissions
- Ensure no other app is using the camera
- Try different camera indices by changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Emoji Images Not Loading
- Verify image files are in the same directory as the script
- Check file names match exactly: `smile.jpg`, `plain.png`, `air.jpg`
- Ensure image files are not corrupted

### Detection Issues
- Ensure good lighting on your face
- Keep your face clearly visible in the camera
- Adjust `SMILE_THRESHOLD` if needed
- For hands up detection, make sure your arms are clearly visible

## Technical Details

- Uses OpenCV for camera capture and display
- MediaPipe Pose and FaceMesh for detection
- Real-time RGB conversion and landmark detection

## Dependencies

- `opencv-python` - Computer vision library
- `mediapipe` - Pose and Face Mesh detection
- `numpy` - Numerical computing

See `requirements.txt` for installation and `requirements-lock.txt` for pinned versions.

## License

MIT License - see LICENSE file for details.
