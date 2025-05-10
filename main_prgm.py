import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import sys

# === Voice Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('voice', engine.getProperty('voices')[1].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# === Get Exercise Type from Argument ===
if len(sys.argv) > 1:
    exercise = sys.argv[1]
else:
    print("❌ No exercise provided. Exiting.")
    exit()

speak(f"Starting {exercise} exercises.")

# === Mediapipe Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
JOINTS = {name: idx for idx, name in enumerate(mp_pose.PoseLandmark)}

# === Pose Dictionary ===
exercise_poses = {
    "Stroke Rehabilitation": [
        "Lift Left Arm", "Lift Right Arm", "Reach Forward",
        "Turn Head Left", "Turn Head Right", "Touch Knees"
    ],
    "Arthritis": [
        "Open and Close Hands", "Wrist Circles", "Shoulder Shrug",
        "Neck Roll", "Finger Stretch", "Ankle Circles"
    ],
    "Sports Injury Recovery": [
        "Step Forward", "Raise Right Knee", "Raise Left Knee",
        "Side Leg Raise", "Toe Touch", "Hamstring Stretch"
    ],
    "Post Surgery Rehab": [
        "Move Ankles Up and Down", "Slide Heel Forward", "Tighten Thigh Muscle",
        "Lift Straight Leg", "Elbow Bend", "Shoulder Lift"
    ],
    "General Flexibility": [
        "Neck Tilt", "Bend to Touch Toes", "Side Reach",
        "Overhead Arm Stretch", "Hip Rotation", "Twist Upper Body"
    ]
}

# === Webcam + Pose Detection ===
cap = cv2.VideoCapture(0)
pose_match_threshold = 25
frame_rate = 5
accuracy_total = 0

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.6) as pose:
    for pose_name in exercise_poses[exercise]:
        print(f"\n👉 Now: {pose_name} (for 20 seconds)")
        speak(f"Now, please perform the pose: {pose_name}")
        pose_duration = 20
        start_time = time.time()
        matched_frames = 0
        total_checked = 0
        last_pose_landmarks = None
        frame_count = 0

        while time.time() - start_time < pose_duration:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width = image.shape[:2]
                for idx, lm in enumerate(landmarks):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    if 0 <= cx < width and 0 <= cy < height:
                        joint_name = mp_pose.PoseLandmark(idx).name
                        cv2.putText(image, joint_name, (cx, cy),
                                    cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 255), 1)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if frame_count % frame_rate == 0:
                    current_pose = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    if last_pose_landmarks is not None and current_pose.shape == last_pose_landmarks.shape:
                        diff = np.linalg.norm(current_pose - last_pose_landmarks)
                        if diff < 0.05:
                            matched_frames += 1
                    last_pose_landmarks = current_pose
                    total_checked += 1

            cv2.putText(image, f"{exercise} | Pose: {pose_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Exercise Guidance", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting on user request.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        pose_accuracy = (matched_frames / total_checked) * 100 if total_checked > 0 else 0
        accuracy_total += pose_accuracy
        print(f"✅ Pose held accuracy: {pose_accuracy:.2f}%")
        speak(f"Pose accuracy: {pose_accuracy:.0f} percent.")

final_accuracy = accuracy_total / len(exercise_poses[exercise])
final_msg = f"\n🎯 Final Exercise Accuracy: {final_accuracy:.2f}%"

if final_accuracy >= 70:
    final_msg += "\n🎉 You did really well! Congratulations!"
    speak("Congratulations! You did really well!")
else:
    final_msg += "\n👍 Keep practicing and you'll improve!"
    speak("Keep practicing and you'll improve!")

print(final_msg)

with open("last_accuracy.txt", "w") as f:
    f.write(f"{final_accuracy:.2f}")

cap.release()
cv2.destroyAllWindows()
print("✅ Exercise Complete. Great job!")
