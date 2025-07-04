import numpy as np
import cv2
import mediapipe as mp
import math
import pandas as pd
from datetime import datetime

class PoseAnalyzer:
    """
    A class to analyze human pose and provide feedback for exercises
    """
    
    def __init__(self):
        # Initialize MediaPipe pose detection with optimized settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Reduced from 1 to 0 for better performance
            enable_segmentation=False,
            min_detection_confidence=0.7,  # Increased for more stable detection
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Squat tracking variables
        self.squat_stage = "up"  # "up" or "down"
        self.squat_count = 0
        self.last_angle = 180
        self.feedback_message = "Ready to start!"
        self.accuracy_score = 0
        self.frame_skip_counter = 0
        
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points using cosine rule
        
        Args:
            point1, point2, point3: Landmark points [x, y]
            
        Returns:
            angle: Angle in degrees
        """
        try:
            # Convert to numpy arrays
            a = np.array(point1)  # First point
            b = np.array(point2)  # Middle point (vertex)
            c = np.array(point3)  # Third point
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle using dot product
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            
            # Ensure cosine is within valid range [-1, 1]
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            # Convert to degrees
            angle = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle)
            
            return angle_degrees
            
        except Exception as e:
            return 180  # Return neutral angle if calculation fails
    
    def detect_squat(self, landmarks):
        """
        Detect squat movement and count repetitions
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            dict: Contains count, stage, feedback, and accuracy
        """
        try:
            # Get key landmarks for squat analysis
            # Left leg landmarks
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Right leg landmarks
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate knee angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Use average of both knees for more stable detection
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            self.last_angle = avg_knee_angle
            
            # Squat detection logic with hysteresis to prevent flickering
            if avg_knee_angle > 160:  # Standing position
                if self.squat_stage == "down":
                    # Complete squat - count it!
                    self.squat_count += 1
                    self.feedback_message = "Great squat! 💪"
                    self.accuracy_score = min(100, self.accuracy_score + 10)
                self.squat_stage = "up"
                
            elif avg_knee_angle < 90:  # Deep squat position
                if self.squat_stage == "up":
                    self.feedback_message = "Perfect depth! Now stand up!"
                    self.accuracy_score = min(100, self.accuracy_score + 5)
                self.squat_stage = "down"
                
            elif 90 <= avg_knee_angle <= 160:  # Transition zone
                if self.squat_stage == "up" and avg_knee_angle < 140:
                    self.feedback_message = "Keep going down! 🔽"
                elif self.squat_stage == "down" and avg_knee_angle > 120:
                    self.feedback_message = "Push up! 🔼"
            
            # Provide form feedback (less frequently to avoid spam)
            if avg_knee_angle > 140 and self.squat_stage == "down":
                self.feedback_message = "Squat deeper for better results! 📉"
                self.accuracy_score = max(0, self.accuracy_score - 1)
            
            return {
                'count': self.squat_count,
                'stage': self.squat_stage,
                'angle': round(avg_knee_angle, 1),
                'feedback': self.feedback_message,
                'accuracy': min(100, max(0, self.accuracy_score))
            }
            
        except Exception as e:
            return {
                'count': self.squat_count,
                'stage': self.squat_stage,
                'angle': 180,
                'feedback': "Position yourself in camera view",
                'accuracy': 0
            }
    
    def process_frame(self, frame, activity_mode="Free Pose"):
        """
        Process video frame for pose detection with performance optimizations
        
        Args:
            frame: Input video frame
            activity_mode: "Free Pose" or "Squat Counter"
            
        Returns:
            processed_frame: Frame with pose overlay
            pose_data: Dictionary with pose analysis results
        """
        try:
            # Resize frame for better performance
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            # Convert back to BGR for display
            output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            pose_data = {
                'landmarks_detected': False,
                'squat_data': None
            }
            
            if results.pose_landmarks:
                pose_data['landmarks_detected'] = True
                
                # Draw pose landmarks with optimized settings
                self.mp_drawing.draw_landmarks(
                    output_frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Analyze squats if in squat mode
                if activity_mode == "Squat Counter":
                    squat_data = self.detect_squat(results.pose_landmarks.landmark)
                    pose_data['squat_data'] = squat_data
                    
                    # Add text overlay with squat info
                    cv2.putText(output_frame, f"Reps: {squat_data['count']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(output_frame, f"Stage: {squat_data['stage'].upper()}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(output_frame, f"Angle: {squat_data['angle']}°", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return output_frame, pose_data
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame, {'landmarks_detected': False, 'squat_data': None}
    
    def reset_counter(self):
        """Reset squat counter and related variables"""
        self.squat_count = 0
        self.squat_stage = "up"
        self.accuracy_score = 0
        self.feedback_message = "Ready to start!"
    
    def save_workout_data(self, reps, accuracy):
        """
        Save workout data to CSV file
        
        Args:
            reps: Number of repetitions completed
            accuracy: Accuracy score percentage
        """
        try:
            workout_data = {
                'date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'exercise': ['Squats'],
                'reps': [reps],
                'accuracy': [accuracy]
            }
            
            df = pd.DataFrame(workout_data)
            
            # Try to append to existing file, create new if doesn't exist
            try:
                existing_df = pd.read_csv('workout_history.csv')
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv('workout_history.csv', index=False)
            except FileNotFoundError:
                df.to_csv('workout_history.csv', index=False)
                
        except Exception as e:
            print(f"Error saving workout data: {e}")

def get_pose_landmarks_info():
    """
    Return information about MediaPipe pose landmarks
    """
    return {
        'total_landmarks': 33,
        'key_points': [
            'Nose', 'Eyes', 'Ears', 'Mouth',
            'Shoulders', 'Elbows', 'Wrists', 'Hands',
            'Hips', 'Knees', 'Ankles', 'Feet'
        ],
        'connections': 'Full body skeleton with 32 connections'
    }