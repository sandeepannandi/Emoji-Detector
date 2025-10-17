import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


SMILE_THRESHOLD = 0.35
HEAD_TILT_THRESHOLD = 0.08  # Reduced threshold for more sensitive detection
JAWLINE_FINGER_DISTANCE = 0.15  # Increased distance for easier detection
BOWL_MOVEMENT_THRESHOLD = 0.05  # Threshold for bowl-shaped hand movement
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

    # Variables for tracking hand movement for bowl detection
    hand_positions_history = []
    MAX_HISTORY = 10

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

        # Check for 67 Clash Royale (bowl-shaped hand movement) - highest priority
        bowl_detected = False
        
        if results_hands.multi_hand_landmarks:
            # Check for bowl-shaped movement with both hands
            if len(results_hands.multi_hand_landmarks) >= 2:
                hand_centers = []
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    # Calculate hand center (middle of palm)
                    wrist = hand_landmarks.landmark[0]
                    middle_finger_mcp = hand_landmarks.landmark[9]
                    hand_center_x = (wrist.x + middle_finger_mcp.x) / 2
                    hand_center_y = (wrist.y + middle_finger_mcp.y) / 2
                    hand_centers.append((hand_center_x, hand_center_y))
                
                # Store hand positions for movement analysis
                if len(hand_centers) == 2:
                    hand_positions_history.append(hand_centers)
                    if len(hand_positions_history) > MAX_HISTORY:
                        hand_positions_history.pop(0)
                    
                    # Check for bowl movement pattern (hands moving up and down together)
                    if len(hand_positions_history) >= 5:
                        recent_positions = hand_positions_history[-5:]
                        y_movements = []
                        for pos in recent_positions:
                            avg_y = (pos[0][1] + pos[1][1]) / 2
                            y_movements.append(avg_y)
                        
                        # Check for alternating up-down movement
                        movement_variance = np.var(y_movements)
                        if movement_variance > BOWL_MOVEMENT_THRESHOLD:
                            bowl_detected = True

        # Check for hands up (second priority)
        hands_up_detected = False
        if results_pose.pose_landmarks and not bowl_detected:
            landmarks = results_pose.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                hands_up_detected = True
        
        # Check for jawline detection (third priority)
        jawline_detected = False
        if not bowl_detected and not hands_up_detected:
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
                        # For now, just trigger on head tilt (we can add finger detection later)
                        jawline_detected = True
                        
                        # Optional: Check if finger is near jawline area for bonus detection
                        if results_hands.multi_hand_landmarks:
                            for hand_landmarks in results_hands.multi_hand_landmarks:
                                # Get index finger tip
                                index_tip = hand_landmarks.landmark[8]
                                
                                # Define jawline area around chin and jaw
                                jawline_area_x = chin.x
                                jawline_area_y = chin.y
                                
                                # Calculate distance from finger to jawline area
                                finger_distance = ((index_tip.x - jawline_area_x)**2 + 
                                                 (index_tip.y - jawline_area_y)**2)**0.5
                                
                                # Also check if finger is in the general jaw area (more flexible)
                                jaw_area_x_min = min(nose_tip.x, chin.x) - 0.1
                                jaw_area_x_max = max(nose_tip.x, chin.x) + 0.1
                                jaw_area_y_min = chin.y - 0.05
                                jaw_area_y_max = chin.y + 0.1
                                
                                # Check if finger is in jaw area
                                in_jaw_area = (jaw_area_x_min <= index_tip.x <= jaw_area_x_max and 
                                             jaw_area_y_min <= index_tip.y <= jaw_area_y_max)
                                
                                if finger_distance < JAWLINE_FINGER_DISTANCE or in_jaw_area:
                                    # Finger is near jawline - this confirms the gesture
                                    break
        
        # Check facial expression for remaining states
        if not bowl_detected and not hands_up_detected and not jawline_detected:
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
        if bowl_detected:
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
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[362]
                eye_line_slope = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x) if (right_eye.x - left_eye.x) != 0 else 0
                head_tilt_angle = abs(eye_line_slope)
                debug_text += f' | Tilt: {head_tilt_angle:.3f}'
                break
        
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
