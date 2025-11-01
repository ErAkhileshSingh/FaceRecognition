import cv2
import os
from dotenv import load_dotenv
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import streamlit as st
from pymongo import MongoClient
from mediapipe.framework.formats import landmark_pb2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av # For handling video frames


load_dotenv()
# --- Constants ---
# IMPORTANT: Place your model files in the same folder as this script
LANDMARKER_MODEL_PATH = 'face_landmarker.task'
RECOGNITION_MODEL_PATH = 'mobilenet_v3_small.tflite'
RECOGNITION_THRESHOLD = 0.6 

# Blink Detection Constants
EAR_THRESHOLD = 0.20 
EAR_CONSEC_FRAMES = 2 
LIVENESS_TIMEOUT = 5 

# MediaPipe Eye Indices
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDXS = [263, 387, 385, 362, 380, 373]

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

# --- Streamlit Session State Initialization ---
# This is how we remember things in Streamlit
if "register_name" not in st.session_state:
    st.session_state.register_name = None
if "status_message" not in st.session_state:
    st.session_state.status_message = ""
if "status_color" not in st.session_state:
    st.session_state.status_color = "black"

# --- Database & Model Loading (Cached) ---

# @st.cache_resource tells Streamlit to run this function ONCE and store the result.
@st.cache_resource
def get_db_connection():
    """Connects to MongoDB and returns the client and collection."""
    try:
        # PASTE YOUR MONGODB URI LINK HERE
        MONGODB_URI = os.environ.get("MONGODB_URI")
        
        client = MongoClient(MONGODB_URI)
        client.admin.command('ping') 
        db = client.sid
        faces_collection = db["faces"]
        st.success("MongoDB connected successfully.")
        return client, faces_collection
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None, None

@st.cache_resource
def load_models():
    """Loads MediaPipe models once and returns them."""
    try:
        landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=False,
            num_faces=1
        )
        landmarker = FaceLandmarker.create_from_options(landmarker_options)

        embedder_options = ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=RECOGNITION_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            l2_normalize=True
        )
        embedder = ImageEmbedder.create_from_options(embedder_options)
        
        return landmarker, embedder
    except Exception as e:
        st.error(f"Failed to load models: {e}. Make sure model files are in the same folder.")
        return None, None

# @st.cache_data tells Streamlit to run this function and cache the *data* it returns.
# We can clear this cache later when we add a new face.
@st.cache_data
def load_known_faces_from_db(_faces_collection):
    """Loads known faces from the database."""
    if _faces_collection is not None:
        faces = _faces_collection.find()
        temp_embeddings = []
        temp_names = []
        for face in faces:
            temp_names.append(face['name'])
            temp_embeddings.append(np.array(face['embedding'], dtype=np.float32))
        return {"names": temp_names, "embeddings": temp_embeddings}
    return {"names": [], "embeddings": []}

# --- Helper Functions ---

def calculate_ear(landmarks: list, eye_idxs: list, frame_shape: tuple) -> float:
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    try:
        def get_coord(lm_idx):
            lm = landmarks[lm_idx]
            return np.array([lm.x * frame_shape[1], lm.y * frame_shape[0]])

        p1, p2, p3, p4, p5, p6 = [get_coord(idx) for idx in eye_idxs]
        
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)

        if C == 0: return 0.3 # Avoid division by zero
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception:
        return 0.3

def get_embedding_from_crop(face_crop_np: np.ndarray, _embedder) -> np.ndarray:
    """Generates an embedding from a cropped face image."""
    if face_crop_np.size == 0 or _embedder is None:
        return None
    try:
        face_crop_rgb = cv2.cvtColor(face_crop_np, cv2.COLOR_BGR2RGB)
        face_crop_rgb = np.ascontiguousarray(face_crop_rgb)

        mp_face_crop = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=face_crop_rgb
        )
        embedding_result = _embedder.embed(mp_face_crop)
        if embedding_result.embeddings:
            return np.array(embedding_result.embeddings[0].embedding, dtype=np.float32)
    except Exception:
        return None
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

# --- The Video Transformer Class (Core Logic) ---
# This class processes each frame from the webcam
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self, landmarker, embedder, db_collection, known_faces_data):
        # Store models and data
        self.landmarker = landmarker
        self.embedder = embedder
        self.db_collection = db_collection
        self.known_names = known_faces_data["names"]
        self.known_embeddings = known_faces_data["embeddings"]
        
        # Liveness state variables
        self.ear_consec_counter = 0
        self.liveness_verified = False
        self.last_blink_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """This method is called for every frame."""
        
        # 1. Convert frame to OpenCV format
        annotated_frame = frame.to_ndarray(format="bgr24")
        
        # 2. Flip for selfie view
        annotated_frame = cv2.flip(annotated_frame, 1)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Detect Landmarks
        try:
            detection_result = self.landmarker.detect(mp_image)
        except Exception:
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        # 4. Draw mesh
        annotated_frame = draw_landmarks(annotated_frame, detection_result)
        
        avg_ear = None # Initialize avg_ear

        # 5. Liveness, Recognition, and Registration Logic
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # 5a. Liveness (Blink Detection)
            left_ear = calculate_ear(landmarks, LEFT_EYE_IDXS, annotated_frame.shape)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_IDXS, annotated_frame.shape)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < EAR_THRESHOLD:
                self.ear_consec_counter += 1
            else:
                if self.ear_consec_counter >= EAR_CONSEC_FRAMES:
                    self.liveness_verified = True
                    self.last_blink_time = time.time()
                self.ear_consec_counter = 0

            if self.liveness_verified and (time.time() - self.last_blink_time > LIVENESS_TIMEOUT):
                self.liveness_verified = False

            # 5b. Get Bounding Box
            x_min = min([lm.x for lm in landmarks]) * annotated_frame.shape[1]
            y_min = min([lm.y for lm in landmarks]) * annotated_frame.shape[0]
            x_max = max([lm.x for lm in landmarks]) * annotated_frame.shape[1]
            y_max = max([lm.y for lm in landmarks]) * annotated_frame.shape[0]
            
            padding = 20
            x_min = max(0, int(x_min - padding))
            y_min = max(0, int(y_min - padding))
            x_max = min(annotated_frame.shape[1], int(x_max + padding))
            y_max = min(annotated_frame.shape[0], int(y_max + padding))

            if x_min < x_max and y_min < y_max:
                face_crop_bgr = annotated_frame[y_min:y_max, x_min:x_max]
                live_emb = get_embedding_from_crop(face_crop_bgr, self.embedder)

                if live_emb is not None:
                    # 5c. Registration (Check session state)
                    # This is how we get the name from the UI
                    register_name = st.session_state.get("register_name")
                    if register_name:
                        if self.db_collection:
                            self.db_collection.insert_one({
                                "name": register_name,
                                "embedding": live_emb.tolist(),
                                "timestamp": datetime.now()
                            })
                            
                            # Update status message for the UI
                            st.session_state.status_message = f"Successfully registered {register_name}!"
                            st.session_state.status_color = "green"
                            
                            # Reset the request and clear the data cache
                            st.session_state.register_name = None
                            st.cache_data.clear() 
                        else:
                            st.session_state.status_message = "Database not connected. Cannot register."
                            st.session_state.status_color = "red"

                    # 5d. Recognition
                    if self.known_embeddings:
                        similarities = [cosine_similarity(live_emb, known_emb) for known_emb in self.known_embeddings]
                        max_similarity = np.max(similarities) if similarities else 0.0
                        
                        name = "Unknown"
                        confidence = round(max_similarity * 100, 2)

                        if max_similarity > RECOGNITION_THRESHOLD:
                            name = self.known_names[np.argmax(similarities)]
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
                        cv2.putText(annotated_frame, f"{name} ({confidence}%)", (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 6. Draw Liveness Status
        if self.liveness_verified:
            liveness_text = "Liveness: Verified"
            liveness_color = (0, 255, 0) # Green
        else:
            liveness_text = "Liveness: NOT Verified (Please Blink)"
            liveness_color = (0, 0, 255) # Red
        
        cv2.putText(annotated_frame, liveness_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveness_color, 2)
        
        if avg_ear is not None: # Only show EAR if a face was detected
             cv2.putText(annotated_frame, f"EAR: {avg_ear:.2f}", (500, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # 7. Return the annotated frame
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- Main Streamlit App UI ---
def main():
    st.set_page_config(page_title="Face Recognition with Liveness", layout="wide")
    st.title("Face Recognition + Liveness (Blink to Register)")
    st.write("This app uses your webcam to verify liveness (by blinking) and recognize known faces.")

    # Load models and DB
    client, faces_collection = get_db_connection()
    landmarker, embedder = load_models()
    
    if client is None or faces_collection is None:
        st.error("Application cannot start. Please check MongoDB connection.")
        st.stop()
    
    if not landmarker or not embedder:
        st.error("Application cannot start. Please check model files.")
        st.stop()
        
    known_faces_data = load_known_faces_from_db(faces_collection)
    st.info(f"Loaded {len(known_faces_data['names'])} known faces from the database.")

    # --- UI Controls in Sidebar ---
    st.sidebar.title("Register New Face")
    name_input = st.sidebar.text_input("Enter name:", key="name_input")
    
    if st.sidebar.button("Register This Face", key="register_button"):
        if name_input:
            # This sets the session state, which the VideoTransformer will see
            st.session_state.register_name = name_input
            st.session_state.status_message = f"Requesting capture for {name_input}... Please look at the camera."
            st.session_state.status_color = "blue"
        else:
            st.session_state.status_message = "Please enter a name to register."
            st.session_state.status_color = "red"

    # Display the status message
    status_text = st.sidebar.empty()
    status_text.markdown(f"<span style='color:{st.session_state.status_color};'>{st.session_state.status_message}</span>", unsafe_allow_html=True)

    # --- Video Stream ---
    # We pass the loaded models and data to our VideoTransformer
    # The lambda function is a trick to pass arguments to the class constructor
    webrtc_streamer(
        key="face-recognition-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}), # Helps with connection
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=lambda: FaceRecognitionTransformer(
            landmarker=landmarker,
            embedder=embedder,
            db_collection=faces_collection,
            known_faces_data=known_faces_data
        ),
        async_processing=True,
    )
    
    # A button to clear the status message
    if st.sidebar.button("Clear Status"):
        st.session_state.status_message = ""
        st.session_state.status_color = "black"
        # This re-runs the script, which will update the status text
        st.experimental_rerun()

if __name__ == "__main__":
    main()
