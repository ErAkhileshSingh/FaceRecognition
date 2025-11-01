import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from pymongo import MongoClient

# --- Constants ---
LANDMARKER_MODEL_PATH = r'F:\New folder\python\FaceDetection\face_landmarker.task'
RECOGNITION_MODEL_PATH = r'F:\New folder\python\FaceDetection\mobilenet_v3_small.tflite'
WEBCAM_ID = 0
RECOGNITION_THRESHOLD = 0.6 

# --- NEW: Blink Detection Constants ---
# Paper: "Eye Aspect Ratio (EAR)" by Soukupová and Čech (2016)
EAR_THRESHOLD = 0.20          # Eye Aspect Ratio threshold (is se kam matlab aankh band hai)
EAR_CONSEC_FRAMES = 2         # Kitne frame tak aankh band rehne par "blink" maana jayega
LIVENESS_TIMEOUT = 5          # (Seconds) Blink ke baad kitni der tak 'Verified' dikhega

# MediaPipe 478-landmark model ke specific eye indices
# (P1, P2, P3, P4, P5, P6) format mein
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDXS = [263, 387, 385, 362, 380, 373]

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Initialize MongoDB ---
# --- Initialize MongoDB ---
try:
    # APNA POORA URI LINK YAHAN PASTE KAREIN
    # I've replaced <password> with the one from your other scripts.
    YOUR_URI_LINK = "mongodb+srv://sammyshekhawat5:sammyshekhawat5@cluster0.u3orhz6.mongodb.net/sid?retryWrites=true&w=majority"

    client = MongoClient(YOUR_URI_LINK)
    # Test the connection
    client.admin.command('ping') 
    db = client.sid # Using 'sid' database as specified in your other files
    faces_collection = db.faces
    faces_collection = db["faces"] # Explicitly defining the collection name as "faces"
    print("MongoDB connected successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None # Set client to None if connection fails
    
# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
from mediapipe.framework.formats import landmark_pb2

# --- Global Variables (Models aur Known Faces) ---
landmarker = None
embedder = None
known_embeddings = []
known_names = []
capture_request = None 

# --- NEW: Helper Function for EAR Calculation ---
def calculate_ear(landmarks: list, eye_idxs: list, frame_shape: tuple) -> float:
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    try:
        # Landmark coordinates ko pixel values mein convert karein
        def get_coord(lm_idx):
            lm = landmarks[lm_idx]
            return np.array([lm.x * frame_shape[1], lm.y * frame_shape[0]])

        # Paper ke P1-P6 points
        p1 = get_coord(eye_idxs[0])
        p2 = get_coord(eye_idxs[1])
        p3 = get_coord(eye_idxs[2])
        p4 = get_coord(eye_idxs[3])
        p5 = get_coord(eye_idxs[4])
        p6 = get_coord(eye_idxs[5])

        # Vertical distances
        A = np.linalg.norm(p2 - p6) # |P2 - P6|
        B = np.linalg.norm(p3 - p5) # |P3 - P5|

        # Horizontal distance
        C = np.linalg.norm(p1 - p4) # |P1 - P4|

        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        # print(f"Error calculating EAR: {e}")
        return 0.3 # Default (eye open) value agar error aaye

# --- Helper Functions (Ye models par depend karti hain) ---
def get_embedding_from_crop(face_crop_np: np.ndarray) -> np.ndarray:
    """Generates an embedding from a cropped face image."""
    global embedder
    if face_crop_np.size == 0 or embedder is None:
        return None
    try:
        # Ensure RGB + contiguous memory
        face_crop_rgb = cv2.cvtColor(face_crop_np, cv2.COLOR_BGR2RGB)
        face_crop_rgb = np.ascontiguousarray(face_crop_rgb)

        mp_face_crop = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=face_crop_rgb
        )

        embedding_result = embedder.embed(mp_face_crop)
        if embedding_result.embeddings:
            return np.array(embedding_result.embeddings[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
    return None


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two normalized vectors."""
    return np.dot(v1, v2)

def draw_landmarks(image: np.ndarray, detection_result: mp.tasks.vision.FaceLandmarkerResult):
    """Draws the face landmarks on the image."""
    if not detection_result.face_landmarks:
        return image
    annotated_image = image.copy()
    for face_landmarks in detection_result.face_landmarks:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in face_landmarks
        ])
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
    return annotated_image

# --- Database Functions ---
def load_known_faces_from_db():
    """MongoDB se sabhi chehron ko global variables mein load karta hai."""
    global known_embeddings, known_names
    if client: # Only run if DB connection is successful
        faces = faces_collection.find()
        temp_embeddings = []
        temp_names = []
        for face in faces:
            temp_names.append(face['name'])
            temp_embeddings.append(np.array(face['embedding'], dtype=np.float32))
        
        known_embeddings = temp_embeddings
        known_names = temp_names
        print(f"Loaded {len(known_embeddings)} embeddings from DB.")

# --- Model Loading Function ---
def load_models():
    """MediaPipe models ko global variables mein load karta hai."""
    global landmarker, embedder
    
    print("Loading face landmarker model...")
    try:
        landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=False,
            num_faces=1
        )
        landmarker = FaceLandmarker.create_from_options(landmarker_options)
        print("Face landmarker loaded.")
    except Exception as e:
        print(f"Failed to load landmarker model: {e}")

    print("Loading face recognition model (ImageEmbedder)...")
    try:
        embedder_options = ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=RECOGNITION_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            l2_normalize=True
        )
        embedder = ImageEmbedder.create_from_options(embedder_options)
        print("Face recognition model loaded.")
    except Exception as e:
        print(f"Failed to load ImageEmbedder model: {e}")

# --- Video Streaming Generator ---
def generate_frames():
    """Webcam se frames capture karta hai, recognition aur liveness chalata hai, aur stream karta hai."""
    global landmarker, embedder, capture_request
    
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {WEBCAM_ID}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # --- NEW: Blink detection state variables ---
    ear_consec_counter = 0      # Frame counter (kitni der se aankh band hai)
    liveness_verified = False   # Kya liveness check pass hua hai?
    last_blink_time = 0         # Aakhri blink kab hua tha (timeout ke liye)
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            detection_result = landmarker.detect(mp_image)
        except Exception as e:
            continue

        annotated_frame = frame.copy()
        annotated_frame = draw_landmarks(annotated_frame, detection_result)

        # Recognition, Registration, aur Liveness logic
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # --- NEW: Calculate EAR ---
            left_ear = calculate_ear(landmarks, LEFT_EYE_IDXS, frame.shape)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_IDXS, frame.shape)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # --- NEW: Liveness Logic ---
            if avg_ear < EAR_THRESHOLD:
                ear_consec_counter += 1 # Aankh band hai, counter badhao
            else:
                # Agar aankh khuli hai, check karo ki kya yeh blink tha
                if ear_consec_counter >= EAR_CONSEC_FRAMES:
                    # Haan, yeh ek poora blink tha
                    liveness_verified = True
                    last_blink_time = time.time()
                ear_consec_counter = 0 # Counter reset karo

            # Liveness status ko timeout ke baad reset karo
            if liveness_verified and (time.time() - last_blink_time > LIVENESS_TIMEOUT):
                liveness_verified = False

            # --- Bounding Box (Pehle se tha) ---
            x_min = min([lm.x for lm in landmarks]) * frame.shape[1]
            y_min = min([lm.y for lm in landmarks]) * frame.shape[0]
            x_max = max([lm.x for lm in landmarks]) * frame.shape[1]
            y_max = max([lm.y for lm in landmarks]) * frame.shape[0]
            
            padding = 20
            x_min = max(0, int(x_min - padding))
            y_min = max(0, int(y_min - padding))
            x_max = min(frame.shape[1], int(x_max + padding))
            y_max = min(frame.shape[0], int(y_max + padding))

            if x_min < x_max and y_min < y_max:
                face_crop_rgb = rgb_frame[y_min:y_max, x_min:x_max]
                live_emb = get_embedding_from_crop(face_crop_rgb)
                
                if live_emb is not None:
                    # --- Naya Chehra Register Karein (Agar request ho) ---
                    if capture_request:
                        if client: # Only attempt to save if DB is connected
                            name_to_save = capture_request['name']
                            embedding_to_save = live_emb.tolist() 
                            
                            faces_collection.insert_one({
                                "name": name_to_save,
                                "embedding": embedding_to_save,
                                "timestamp": datetime.now()
                            })
                            print(f"Successfully registered {name_to_save}")
                            load_known_faces_from_db() 
                            capture_request = None 
                        else:
                            print("Cannot register face: Database not connected.")
                            capture_request = None # Reset request if DB is down
                    
                    # --- Chehra Pehchaanein ---
                    if known_embeddings:
                        similarities = [cosine_similarity(live_emb, known_emb) for known_emb in known_embeddings]
                        max_similarity = np.max(similarities) if similarities else 0.0
                        
                        name = "Unknown"
                        confidence = round(max_similarity * 100, 2)

                        if max_similarity > RECOGNITION_THRESHOLD:
                            name = known_names[np.argmax(similarities)]
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(annotated_frame, f"{name} ({confidence}%)", (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- NEW: Liveness Status ko Screen par Draw Karein ---
        if liveness_verified:
            liveness_text = "Liveness: Verified"
            liveness_color = (0, 255, 0) # Green
        else:
            liveness_text = "Liveness: NOT Verified (Please Blink)"
            liveness_color = (0, 0, 255) # Red
        
        cv2.putText(annotated_frame, liveness_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveness_color, 2)
        
        if 'avg_ear' in locals(): # Agar chehra detect hua hai toh hi EAR dikhayein
            cv2.putText(annotated_frame, f"EAR: {avg_ear:.2f}", (500, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # Frame ko JPEG mein encode karein
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error encoding frame: {e}")
            
    cap.release()

# --- Flask Routes ---

@app.route('/')
def index():
    """Home page render karta hai."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. `generate_frames` function ko call karta hai."""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])

# --- YEH FUNCTION WEB UI KE LIYE HAI ---
@app.route('/register', methods=['POST'])
def register_face():
    """Web UI se request handle karta hai, video loop ke liye flag set karta hai."""
    global capture_request
    data = request.get_json()
    name = data.get('name')
    embedding = data.get('embedding')
    
    if not name:
        return jsonify({"status": "error", "message": "Name is required"}), 400
    
    # Yeh sirf flag set karta hai, generate_frames() baki kaam karega
    capture_request = {"name": name} 
    print(f"Capture request received for: {name}")
    
    return jsonify({"status": "success", "message": f"Capturing face for {name}..."})

# --- !!!!! NEW FUNCTION FOR POSTMAN !!!!! ---
@app.route('/register_manual', methods=['POST'])
def register_face_manual():
    """
    POSTMAN se data register karne ke liye.
    Yeh function JSON mein 'name' aur 'embedding' expect karta hai.
    """
    if not client:
        return jsonify({"status": "error", "message": "Database not connected"}), 500

    data = request.get_json()
    name = data.get('name')
    embedding = data.get('embedding') # Yeh ek list honi chahiye

    # --- Error Checking ---
    if not name:
        return jsonify({"status": "error", "message": "Name is required"}), 400
    if not embedding:
        return jsonify({"status": "error", "message": "Embedding is required"}), 400
    if not isinstance(embedding, list):
        return jsonify({"status": "error", "message": "Embedding must be a list of numbers"}), 400

    # --- Data ko DB mein Save Karein ---
    try:
        faces_collection.insert_one({
            "name": name,
            "embedding": embedding, # Embedding ko list ki tarah save karein
            "timestamp": datetime.now()
        })
        
        print(f"Successfully registered {name} manually from Postman.")
        
        # Known faces ko reload karein taaki video stream isko pehchan sake
        load_known_faces_from_db() 
        
        return jsonify({
            "status": "success", 
            "message": f"Successfully registered {name} with {len(embedding)} embedding values."
        }), 201

    except Exception as e:
        print(f"Error during manual registration: {e}")
        return jsonify({"status": "error", "message": f"An error occurred: {e}"}), 500


# --- Main ---
if __name__ == "__main__":
    load_models()
    load_known_faces_from_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
