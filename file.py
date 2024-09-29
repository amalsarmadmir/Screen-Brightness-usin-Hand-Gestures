import cv2
import mediapipe as mp
import screen_brightness_control as sbc

def calculate_brightness(landmark1, landmark2):
    # Euclidean distance between the thumb tip and index finger tip
    distance = ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5
    # Convert the distance into a brightness value (0-100)
    brightness = int(distance * 500)  # Adjust scaling factor as necessary
    return min(max(brightness, 0), 100)  # Clamp between 0 and 100


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(image_rgb)
    
    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmark positions for the thumb tip (4) and index finger tip (8)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate brightness based on the distance between thumb and index finger
            brightness = calculate_brightness(thumb_tip, index_tip)
            
            # Set the screen brightness
            sbc.set_brightness(brightness)
            
            # Display the brightness value on the image
            cv2.putText(image, f'Brightness: {brightness}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Hand Gesture Brightness Control', image)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
