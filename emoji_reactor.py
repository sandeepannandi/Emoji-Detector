import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


SMILE_THRESHOLD = 0.35
HEAD_TILT_THRESHOLD = 0.12  # Increased threshold for more pronounced head tilt
JAWLINE_FINGER_DISTANCE = 0.2  # Increased distance for easier finger detection
BOWL_MOVEMENT_THRESHOLD = 0.03  # Threshold for opposite hand movement
HAND_PROXIMITY_THRESHOLD = 0.3  # Threshold for hands being near abdomen/shoulder
OPPOSITE_CORR_THRESHOLD = -0.2  # Negative correlation threshold for opposite movement
MIN_OPPOSITE_AMPLITUDE = 0.04   # Min normalized vertical range per hand in window
STICKY_FRAMES_67 = 12           # Keep 67 state active this many frames to reduce flicker
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    jawline_emoji = cv2.imread("jawline.png")
    clash_royale_emoji = cv2.imread("67clashroyale.webp")

    if smiling_emoji is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.jpg not found")
    if jawline_emoji is None:
        raise FileNotFoundError("jawline.png not found")
    if clash_royale_emoji is None:
        raise FileNotFoundError("67clashroyale.webp not found")

    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    jawline_emoji = cv2.resize(jawline_emoji, EMOJI_WINDOW_SIZE)
    clash_royale_emoji = cv2.resize(clash_royale_emoji, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.jpg (smiling face)")
    print("- plain.png (straight face)")
    print("- air.jpg (hands up)")
    print("- jawline.png (jawline flex)")
    print("- 67clashroyale.webp (67 meme)")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  Raise hands above shoulders for hands up")
print("  Smile for smiling emoji")
print("  Straight face for neutral emoji")
print("  Tilt head left/right + finger on jawline for jawline flex")
print("  Move both hands up/down in bowl shape for 67 meme")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    # Variables for tracking hand movement for 67 detection
    hand_positions_history = []  # legacy window of paired positions
    hand1_y_history = []
    hand2_y_history = []
    MAX_HISTORY = 20
    clash_sticky_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"

        # Process all MediaPipe solutions first
        results_pose = pose.process(image_rgb)
        results_face = face_mesh.process(image_rgb)
        results_hands = hands.process(image_rgb)

        # Check for 67 Clash Royale (opposite hand movement) - highest priority
        clash_royale_detected = False
        
        if results_hands.multi_hand_landmarks and len(results_hands.multi_hand_landmarks) >= 2 and results_pose.pose_landmarks:
            # Get both hands
            hand1_landmarks = results_hands.multi_hand_landmarks[0]
            hand2_landmarks = results_hands.multi_hand_landmarks[1]
            
            # Wrist positions
            hand1_wrist = hand1_landmarks.landmark[0]
            hand2_wrist = hand2_landmarks.landmark[0]
            
            # Torso landmarks
            pose_landmarks = results_pose.pose_landmarks.landmark
            left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Torso geometry
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            torso_height = max(1e-6, abs(hip_center_y - shoulder_center_y))
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            torso_center_x = (shoulder_center_x + hip_center_x) / 2
            torso_center_y = (shoulder_center_y + hip_center_y) / 2
            
            # Proximity to torso center
            hand1_distance = ((hand1_wrist.x - torso_center_x)**2 + (hand1_wrist.y - torso_center_y)**2)**0.5
            hand2_distance = ((hand2_wrist.x - torso_center_x)**2 + (hand2_wrist.y - torso_center_y)**2)**0.5
            near_torso = hand1_distance < HAND_PROXIMITY_THRESHOLD and hand2_distance < HAND_PROXIMITY_THRESHOLD
            
            # Normalize Y by torso height to be scale-invariant
            hand1_norm_y = (hand1_wrist.y - torso_center_y) / torso_height
            hand2_norm_y = (hand2_wrist.y - torso_center_y) / torso_height
            
            # Update histories
            hand1_y_history.append(hand1_norm_y)
            hand2_y_history.append(hand2_norm_y)
            if len(hand1_y_history) > MAX_HISTORY:
                hand1_y_history.pop(0)
            if len(hand2_y_history) > MAX_HISTORY:
                hand2_y_history.pop(0)
            
            # Analyze recent window
            WINDOW = 10
            if near_torso and len(hand1_y_history) >= WINDOW and len(hand2_y_history) >= WINDOW:
                h1 = np.array(hand1_y_history[-WINDOW:])
                h2 = np.array(hand2_y_history[-WINDOW:])
                dy1 = np.diff(h1)
                dy2 = np.diff(h2)
                amp1 = float(np.ptp(h1))
                amp2 = float(np.ptp(h2))
                corr = 0.0
                if np.std(dy1) > 1e-6 and np.std(dy2) > 1e-6:
                    corr = float(np.corrcoef(dy1, dy2)[0, 1])
                
                # Opposite movement with enough amplitude
                if corr <= OPPOSITE_CORR_THRESHOLD and amp1 >= MIN_OPPOSITE_AMPLITUDE and amp2 >= MIN_OPPOSITE_AMPLITUDE:
                    clash_royale_detected = True
                    clash_sticky_counter = STICKY_FRAMES_67
                elif clash_sticky_counter > 0:
                    clash_royale_detected = True
                    clash_sticky_counter -= 1
        else:
            # Decay sticky if hands/pose not available
            if clash_sticky_counter > 0:
                clash_royale_detected = True
                clash_sticky_counter -= 1

        # Check for hands up (second priority)
        hands_up_detected = False
        if results_pose.pose_landmarks and not clash_royale_detected:
            landmarks = results_pose.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                hands_up_detected = True
        
        # Check for jawline detection (third priority)
        jawline_detected = False
        if not clash_royale_detected and not hands_up_detected:
            # Check for head tilt using face mesh (more accurate)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # Get key facial points
                    nose_tip = face_landmarks.landmark[1]  # Nose tip
                    left_eye = face_landmarks.landmark[33]  # Left eye corner
                    right_eye = face_landmarks.landmark[362]  # Right eye corner
                    chin = face_landmarks.landmark[175]  # Chin point
                    
                    # Calculate head tilt using eye line
                    eye_center_x = (left_eye.x + right_eye.x) / 2
                    eye_center_y = (left_eye.y + right_eye.y) / 2
                    
                    # Calculate tilt angle
                    eye_line_slope = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x) if (right_eye.x - left_eye.x) != 0 else 0
                    head_tilt_angle = abs(eye_line_slope)
                    
                    # Check if head is tilted significantly
                    if head_tilt_angle > HEAD_TILT_THRESHOLD:
                        # Check if finger is near jawline area (required for jawline detection)
                        finger_near_jawline = False
                        if results_hands.multi_hand_landmarks:
                            for hand_landmarks in results_hands.multi_hand_landmarks:
                                # Get index finger tip
                                index_tip = hand_landmarks.landmark[8]
                                
                                # Define jawline area around chin and jaw (more generous area)
                                jawline_area_x = chin.x
                                jawline_area_y = chin.y
                                
                                # Calculate distance from finger to jawline area
                                finger_distance = ((index_tip.x - jawline_area_x)**2 + 
                                                 (index_tip.y - jawline_area_y)**2)**0.5
                                
                                # Also check if finger is in the general jaw area (more flexible)
                                jaw_area_x_min = min(nose_tip.x, chin.x) - 0.15
                                jaw_area_x_max = max(nose_tip.x, chin.x) + 0.15
                                jaw_area_y_min = chin.y - 0.08
                                jaw_area_y_max = chin.y + 0.12
                                
                                # Check if finger is in jaw area
                                in_jaw_area = (jaw_area_x_min <= index_tip.x <= jaw_area_x_max and 
                                             jaw_area_y_min <= index_tip.y <= jaw_area_y_max)
                                
                                # Also check if finger is moving near the jawline (dynamic detection)
                                if finger_distance < JAWLINE_FINGER_DISTANCE or in_jaw_area:
                                    finger_near_jawline = True
                                    break
                        
                        # Only trigger jawline detection if BOTH head tilt AND finger near jawline
                        if finger_near_jawline:
                            jawline_detected = True
        
        # Check facial expression for remaining states
        if not clash_royale_detected and not hands_up_detected and not jawline_detected:
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    left_corner = face_landmarks.landmark[291]
                    right_corner = face_landmarks.landmark[61]
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]

                    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
                    
                    if mouth_width > 0:
                        mouth_aspect_ratio = mouth_height / mouth_width
                        if mouth_aspect_ratio > SMILE_THRESHOLD:
                            current_state = "SMILING"
                        else:
                            current_state = "STRAIGHT_FACE"
        
        # Set final state based on detection priority
        if clash_royale_detected:
            current_state = "CLASH_ROYALE_67"
        elif hands_up_detected:
            current_state = "HANDS_UP"
        elif jawline_detected:
            current_state = "JAWLINE_FLEX"
        
        if current_state == "SMILING":
            emoji_to_display = smiling_emoji
            emoji_name = "😊"
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        elif current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        elif current_state == "JAWLINE_FLEX":
            emoji_to_display = jawline_emoji
            emoji_name = "💪"
        elif current_state == "CLASH_ROYALE_67":
            emoji_to_display = clash_royale_emoji
            emoji_name = "67"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Add debug information
        debug_text = f'STATE: {current_state} {emoji_name}'
        finger_status = "No finger"
        clash_status = "No clash"
        
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[362]
                nose_tip = face_landmarks.landmark[1]
                chin = face_landmarks.landmark[175]
                
                eye_line_slope = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x) if (right_eye.x - left_eye.x) != 0 else 0
                head_tilt_angle = abs(eye_line_slope)
                debug_text += f' | Tilt: {head_tilt_angle:.3f}'
                
                # Check finger status
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[8]
                        finger_distance = ((index_tip.x - chin.x)**2 + (index_tip.y - chin.y)**2)**0.5
                        if finger_distance < JAWLINE_FINGER_DISTANCE:
                            finger_status = f"Finger: {finger_distance:.3f}"
                            break
                        else:
                            finger_status = f"Finger: {finger_distance:.3f}"
                
                debug_text += f' | {finger_status}'
                break
        
        # Add clash royale debug info
        if results_hands.multi_hand_landmarks and len(results_hands.multi_hand_landmarks) >= 2 and results_pose.pose_landmarks:
            hand1_wrist = results_hands.multi_hand_landmarks[0].landmark[0]
            hand2_wrist = results_hands.multi_hand_landmarks[1].landmark[0]
            pose_landmarks = results_pose.pose_landmarks.landmark
            left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            torso_height = max(1e-6, abs(hip_center_y - shoulder_center_y))
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            torso_center_x = (shoulder_center_x + hip_center_x) / 2
            torso_center_y = (shoulder_center_y + hip_center_y) / 2
            hand1_distance = ((hand1_wrist.x - torso_center_x)**2 + (hand1_wrist.y - torso_center_y)**2)**0.5
            hand2_distance = ((hand2_wrist.x - torso_center_x)**2 + (hand2_wrist.y - torso_center_y)**2)**0.5
            # Estimate live correlation/amplitude if we have enough samples
            corr_txt = "n/a"
            amp_txt = "n/a"
            WINDOW = 10
            if len(hand1_y_history) >= WINDOW and len(hand2_y_history) >= WINDOW:
                h1 = np.array(hand1_y_history[-WINDOW:])
                h2 = np.array(hand2_y_history[-WINDOW:])
                dy1 = np.diff(h1)
                dy2 = np.diff(h2)
                if np.std(dy1) > 1e-6 and np.std(dy2) > 1e-6:
                    corr_val = float(np.corrcoef(dy1, dy2)[0, 1])
                    corr_txt = f"corr:{corr_val:.2f}"
                amp1 = float(np.ptp(h1))
                amp2 = float(np.ptp(h2))
                amp_txt = f"amp:{amp1:.2f}/{amp2:.2f}"
            clash_status = f"Hands: {hand1_distance:.2f},{hand2_distance:.2f} {corr_txt} {amp_txt} sticky:{clash_sticky_counter}"
        
        debug_text += f' | {clash_status}'
        
        cv2.putText(camera_frame_resized, debug_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
