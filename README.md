# 🏃‍♂️ Real-time Human Pose Estimation Web App

A complete Python-based web application for real-time human pose detection, exercise tracking, and posture feedback using AI and computer vision.

## ✨ Features

- **Real-time Pose Detection**: Uses MediaPipe to detect 33 body landmarks
- **Exercise Tracking**: Smart squat counter with form analysis
- **Voice Feedback**: Audio guidance for better workout experience
- **Posture Analysis**: Real-time feedback on exercise form
- **Accuracy Scoring**: Performance metrics and progress tracking
- **Workout History**: Save and track your exercise sessions
- **Web-based Interface**: Runs entirely in your browser

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera access
- Modern web browser (Chrome, Firefox, Safari)

### Installation

1. **Clone or download the project files**
   ```bash
   # Create a new directory
   mkdir pose-estimation-app
   cd pose-estimation-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open in your default browser
   - If not, go to `http://localhost:8501`

## 📁 Project Structure

```
pose-estimation-app/
├── app.py              # Main Streamlit application
├── utils.py            # Helper functions and pose analysis logic
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── workout_history.csv # Generated workout data (created automatically)
```

## 🎯 How to Use

### Free Pose Mode
1. Select "Free Pose" from the sidebar
2. Allow camera permissions when prompted
3. Stand in front of the camera with your full body visible
4. Watch as the AI detects and draws your pose skeleton in real-time

### Squat Counter Mode
1. Select "Squat Counter" from the sidebar
2. Position yourself 6-8 feet from the camera
3. Ensure your full body is visible in the frame
4. Start performing squats - the AI will:
   - Count your repetitions automatically
   - Provide real-time form feedback
   - Give voice guidance (if enabled)
   - Track your accuracy score

## ⚙️ Settings & Features

### Sidebar Controls
- **Activity Selector**: Choose between "Free Pose" and "Squat Counter"
- **Voice Feedback**: Toggle audio guidance on/off
- **Reset Counter**: Clear current workout session
- **Save Workout**: Store your session data

### Workout Statistics
- **Repetition Counter**: Real-time rep counting
- **Current Stage**: Shows "UP" or "DOWN" position
- **Knee Angle**: Live angle measurement
- **Form Accuracy**: Performance score with progress bar
- **Feedback Messages**: Real-time form corrections

## 🔧 Technical Details

### AI & Computer Vision
- **MediaPipe Pose**: Google's ML framework for pose detection
- **33 Landmarks**: Full body keypoint detection
- **Angle Calculation**: Mathematical analysis using cosine rule
- **Real-time Processing**: Optimized for live video streams

### Exercise Detection Logic
- **Squat Recognition**: Based on knee and hip angle analysis
- **Threshold Detection**: Angles below 90° count as proper depth
- **Stage Tracking**: Monitors "up" and "down" phases
- **Form Feedback**: Provides guidance for better technique

### Performance Optimization
- **Efficient Processing**: Optimized for real-time performance
- **Smart Caching**: Reduces computational overhead
- **Adaptive Quality**: Adjusts based on system capabilities

## 🎨 User Interface

### Main Features
- **Live Camera Feed**: Real-time video with pose overlay
- **Interactive Sidebar**: All controls and statistics
- **Responsive Design**: Works on different screen sizes
- **Clean Layout**: Beginner-friendly interface

### Visual Feedback
- **Pose Skeleton**: Green landmarks and red connections
- **Text Overlays**: Rep count, stage, and angle information
- **Progress Bars**: Visual accuracy indicators
- **Status Indicators**: Connection and detection status

## 🔊 Audio Features

### Voice Feedback System
- **Text-to-Speech**: Uses pyttsx3 for audio guidance
- **Smart Timing**: Provides feedback at appropriate moments
- **Customizable**: Can be toggled on/off
- **Form Corrections**: Audio cues for better technique

### Feedback Messages
- "Great squat!" - Perfect form completion
- "Squat deeper!" - Needs more depth
- "Perfect depth! Now stand up!" - Good bottom position
- "Keep going down!" - Transitioning to squat

## 📊 Data & Analytics

### Workout Tracking
- **Session Data**: Reps, accuracy, timestamp
- **CSV Export**: Automatic data saving
- **Progress Monitoring**: Track improvement over time
- **Performance Metrics**: Detailed accuracy scoring

### Accuracy Scoring
- **Form Analysis**: Based on proper squat mechanics
- **Real-time Updates**: Score changes with performance
- **Feedback Integration**: Linked to voice guidance
- **Progress Tracking**: Shows improvement trends

## 🛠️ Troubleshooting

### Common Issues

**Camera Not Working**
- Allow camera permissions in browser settings
- Check if camera is being used by other applications
- Try refreshing the page

**Pose Not Detected**
- Ensure good lighting conditions
- Stand with full body visible in frame
- Wear contrasting clothing
- Remove cluttered background

**Performance Issues**
- Close unnecessary browser tabs
- Check internet connection
- Reduce video quality if needed
- Restart the application

**Voice Feedback Not Working**
- Check browser audio permissions
- Ensure speakers/headphones are connected
- Try toggling voice feedback off and on

### Optimal Setup
- **Distance**: 6-8 feet from camera
- **Lighting**: Bright, even lighting
- **Background**: Plain, uncluttered
- **Clothing**: Contrasting colors
- **Position**: Full body visible in frame

## 🔒 Privacy & Security

- **Local Processing**: All AI processing happens on your device
- **No Data Upload**: Video never leaves your computer
- **Secure Connection**: Uses encrypted WebRTC protocols
- **Optional Saving**: Workout data saved locally only

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with one click

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new exercise types
- Optimizing performance

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe**: Google's amazing ML framework
- **Streamlit**: For the fantastic web app framework
- **OpenCV**: Computer vision library
- **Community**: All the developers who made this possible

---

**Happy exercising! 🏋️‍♂️💪**

For questions or support, please check the troubleshooting section or create an issue in the project repository.