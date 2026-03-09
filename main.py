from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import cloudinary
import cloudinary.uploader
import cloudinary.api
import io
from PIL import Image
import base64
import time
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
from pydantic import BaseModel
from datetime import datetime
import json
import pickle
from pathlib import Path
import shutil
import asyncio
import queue
import threading
import sys

# Scikit-learn imports for continuous learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import joblib

app = FastAPI(
    title="Waste Detection API with Continuous Learning & Live Detection",
    description="Advanced waste detection with live camera support and continuous learning",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Cloudinary
cloudinary.config(
    cloud_name="dtisam8ot",
    api_key="416996345946976",
    api_secret="dcfIgNOmXE5GkMyXgOAHnMxVeLg",
    secure=True
)

# Data directories
DATA_DIR = Path("training_data")
MODEL_DIR = Path("models")
FEATURES_DIR = Path("features")

# Create directories
for directory in [DATA_DIR, MODEL_DIR, FEATURES_DIR]:
    directory.mkdir(exist_ok=True)

# Pydantic models
class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image
    allow_training: bool = False  # User consent for using data in training
    user_id: Optional[str] = None
    live_detection: bool = False  # Flag for live detection mode
    quality: Optional[int] = 50   # Quality for live detection (1-100)

class DetectedObject(BaseModel):
    label: str
    confidence: float
    box: List[float]
    material: str
    area_percentage: float
    category: str
    features: Optional[Dict[str, float]] = None

class WasteComposition(BaseModel):
    special_waste: float = 0.0
    recyclable: float = 0.0
    residual: float = 0.0

class DetectionResponse(BaseModel):
    success: bool = True
    detected_objects: List[DetectedObject] = []
    overall_category: str = "Unknown"
    overall_confidence: float = 0.0
    waste_composition: WasteComposition = WasteComposition()
    recycling_tips: List[str] = []
    total_objects_detected: int = 0
    cloudinary_url: Optional[str] = None
    cloudinary_public_id: Optional[str] = None
    message: str = ""
    training_consent: bool = False
    dataset_size: int = 0
    processing_time: Optional[float] = None
    live_detection: bool = False

class TrainingRequest(BaseModel):
    min_samples: int = 100
    test_size: float = 0.2
    retrain_yolo: bool = False

class TrainingResponse(BaseModel):
    success: bool
    message: str
    accuracy: Optional[float] = None
    dataset_size: int
    new_classes: List[str] = []

class DatasetInfo(BaseModel):
    total_samples: int
    classes_count: Dict[str, int]
    last_trained: Optional[str] = None
    accuracy: Optional[float] = None

class FrameRequest(BaseModel):
    frame: str  # Base64 encoded frame
    user_id: Optional[str] = None
    allow_training: bool = False
    session_id: Optional[str] = None

class LiveDetectionRequest(BaseModel):
    frames: List[str]  # Multiple base64 frames
    user_id: Optional[str] = None
    allow_training: bool = False
    quality: int = 30

# Initialize Continuous Learning System
class ContinuousLearningSystem:
    def __init__(self):
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.vectorizer = DictVectorizer(sparse=False)
        self.dataset_path = DATA_DIR / "waste_dataset.csv"
        self.model_path = MODEL_DIR / "waste_classifier.pkl"
        self.features_path = FEATURES_DIR / "feature_scaler.pkl"
        self.dataset = None
        self.is_trained = False
        
        # Load existing model and dataset
        self.load_model()
        self.load_dataset()
    
    def load_model(self):
        """Load trained classifier if exists"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.classifier = saved_data['classifier']
                    self.label_encoder = saved_data['label_encoder']
                    self.scaler = saved_data['scaler']
                    self.vectorizer = saved_data['vectorizer']
                print("✅ Loaded existing classifier model")
                self.is_trained = True
            else:
                self.classifier = RandomForestClassifier(
                    n_estimators=50,  # Reduced for faster training
                    max_depth=8,
                    random_state=42
                )
                print("⚠️ Created new classifier - no existing model found")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def load_dataset(self):
        """Load existing dataset"""
        try:
            if self.dataset_path.exists():
                self.dataset = pd.read_csv(self.dataset_path)
                print(f"✅ Loaded dataset with {len(self.dataset)} samples")
            else:
                self.dataset = pd.DataFrame(columns=[
                    'label', 'confidence', 'material', 'category',
                    'area_percentage', 'color_mean_r', 'color_mean_g', 'color_mean_b',
                    'texture_variance', 'aspect_ratio', 'solidity', 'user_id', 'timestamp'
                ])
                print("⚠️ Created new empty dataset")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self.dataset = pd.DataFrame()
    
    def extract_features(self, image_crop: np.ndarray, box_coords: List[float]) -> Dict[str, float]:
        """Extract features from image crop for ML training"""
        features = {}
        
        try:
            if image_crop is not None and image_crop.size > 0:
                # Resize for faster processing
                if image_crop.shape[0] > 100 or image_crop.shape[1] > 100:
                    image_crop = cv2.resize(image_crop, (100, 100))
                
                # Color features
                color_mean = np.mean(image_crop, axis=(0, 1))
                features['color_mean_r'] = float(color_mean[0])
                features['color_mean_g'] = float(color_mean[1])
                features['color_mean_b'] = float(color_mean[2])
                features['color_std'] = float(np.std(image_crop))
                
                # Texture features (simplified)
                gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
                features['texture_variance'] = float(np.var(gray))
                
                # Shape features
                height, width = image_crop.shape[:2]
                features['aspect_ratio'] = float(width / height) if height > 0 else 1.0
                
                # Edge features (simplified)
                if gray.shape[0] > 20 and gray.shape[1] > 20:
                    edges = cv2.Canny(gray, 50, 150)
                    features['edge_density'] = float(np.sum(edges > 0) / edges.size)
                else:
                    features['edge_density'] = 0.0
                
                # Additional features
                features['solidity'] = float(np.mean(gray) / 255.0)
                features['brightness'] = float(np.mean(gray))
                
        except Exception as e:
            # Default features
            features = {
                'color_mean_r': 128.0,
                'color_mean_g': 128.0,
                'color_mean_b': 128.0,
                'color_std': 0.0,
                'texture_variance': 0.0,
                'aspect_ratio': 1.0,
                'edge_density': 0.0,
                'solidity': 0.5,
                'brightness': 128.0
            }
        
        return features
    
    def extract_fast_features(self, image_crop: np.ndarray) -> Dict[str, float]:
        """Extract minimal features for live detection"""
        features = {}
        
        try:
            if image_crop is not None and image_crop.size > 0:
                # Very basic features for speed
                features['color_mean_r'] = float(np.mean(image_crop[:,:,0]))
                features['color_mean_g'] = float(np.mean(image_crop[:,:,1]))
                features['color_mean_b'] = float(np.mean(image_crop[:,:,2]))
                features['brightness'] = float(np.mean(cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)))
        except:
            features = {'color_mean_r': 128.0, 'color_mean_g': 128.0, 'color_mean_b': 128.0, 'brightness': 128.0}
        
        return features
    
    def add_to_dataset(self, detected_objects: List[Dict], image_array: np.ndarray, 
                       original_size: Tuple[int, int], allow_training: bool, user_id: str = None):
        """Add detected objects to training dataset"""
        if not allow_training:
            return
        
        new_samples = []
        timestamp = datetime.now().isoformat()
        
        for obj in detected_objects:
            # Extract image region for feature extraction
            box = obj['box']
            x1 = int(box[0] * original_size[0])
            y1 = int(box[1] * original_size[1])
            x2 = int(box[2] * original_size[0])
            y2 = int(box[3] * original_size[1])
            
            # Minimal padding for speed
            padding = 2
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_array.shape[1], x2 + padding)
            y2 = min(image_array.shape[0], y2 + padding)
            
            if x2 > x1 and y2 > y1:
                image_crop = image_array[y1:y2, x1:x2]
                
                if image_crop.size > 0:
                    features = self.extract_fast_features(image_crop)
                    
                    sample = {
                        'label': obj['label'],
                        'confidence': obj['confidence'],
                        'material': obj['material'],
                        'category': obj['category'],
                        'area_percentage': obj['area_percentage'],
                        'user_id': user_id or 'anonymous',
                        'timestamp': timestamp
                    }
                    sample.update(features)
                    new_samples.append(sample)
        
        if new_samples:
            new_df = pd.DataFrame(new_samples)
            self.dataset = pd.concat([self.dataset, new_df], ignore_index=True)
            
            # Save dataset in background
            threading.Thread(target=self.save_dataset).start()
    
    def save_dataset(self):
        """Save dataset to CSV"""
        try:
            self.dataset.to_csv(self.dataset_path, index=False)
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
    
    def train_classifier(self, test_size: float = 0.2) -> Tuple[bool, float]:
        """Train the classifier on current dataset"""
        try:
            if len(self.dataset) < 30:  # Reduced minimum samples
                print(f"⚠️ Not enough samples for training: {len(self.dataset)}")
                return False, 0.0
            
            # Prepare features and labels
            feature_cols = ['color_mean_r', 'color_mean_g', 'color_mean_b', 'brightness']
            
            if len(self.dataset) > 0:
                X = self.dataset[feature_cols]
                y = self.dataset['label']
                
                # Encode labels
                y_encoded = self.label_encoder.fit_transform(y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42
                )
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train classifier
                self.classifier.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = self.classifier.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Save model
                self.save_model()
                
                self.is_trained = True
                print(f"✅ Classifier trained with accuracy: {accuracy:.2%}")
                return True, accuracy
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False, 0.0
    
    def save_model(self):
        """Save trained model"""
        try:
            model_data = {
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'vectorizer': self.vectorizer,
                'training_date': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def predict_category(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Predict waste category using trained classifier"""
        if not self.is_trained or self.classifier is None:
            return "Unknown", 0.0
        
        try:
            # Prepare feature vector
            feature_cols = ['color_mean_r', 'color_mean_g', 'color_mean_b', 'brightness']
            
            feature_vector = []
            for col in feature_cols:
                feature_vector.append(features.get(col, 0.0))
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Predict
            prediction = self.classifier.predict(feature_vector_scaled)[0]
            probability = np.max(self.classifier.predict_proba(feature_vector_scaled)[0])
            
            label = self.label_encoder.inverse_transform([prediction])[0]
            
            return label, float(probability * 100)
            
        except Exception as e:
            return "Unknown", 0.0

# Initialize systems
learning_system = ContinuousLearningSystem()

# Your waste categories based on 5 classes
WASTE_CATEGORIES = {
    "Recyclable": ["can", "glass bottle", "plastic bottle", "paper"],
    "Residual / Non-Recyclable": ["styrofoam cups"],
    "Special Waste": []  # Add if you have special waste items
}

# Material types
MATERIAL_TYPES = {
    "can": "metal",
    "glass bottle": "glass",
    "plastic bottle": "plastic",
    "paper": "paper",
    "styrofoam cups": "foam"
}

# Load YOLO model
try:
    model = YOLO("runs/detect/waste13/weights/best.pt")
    print("✅ YOLO model loaded successfully")
    
    # Load a lightweight model for live detection
    try:
        live_model = YOLO("yolov8n.pt")  # Nano model for speed
        print("✅ Lightweight YOLO model loaded for live detection")
    except:
        live_model = model  # Fallback to main model
        print("⚠️ Using main model for live detection")
        
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    # Try alternative paths
    model_paths = ["best.pt", "weights/best.pt", "./runs/detect/train/weights/best.pt"]
    model = None
    for path in model_paths:
        try:
            if os.path.exists(path):
                model = YOLO(path)
                print(f"✅ YOLO model loaded from: {path}")
                live_model = model
                break
        except:
            pass
    
    if model is None:
        print("⚠️ Running in debug mode without YOLO model")
        model = None
        live_model = None

# WebSocket connections for real-time video
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.frame_queues = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

def classify_object(label: str) -> str:
    """Classify object into waste category"""
    label_lower = label.lower()
    
    for category, items in WASTE_CATEGORIES.items():
        for item in items:
            if item in label_lower:
                return category
    
    return "Unknown"

def get_material_type(label: str) -> str:
    """Get material type for label"""
    label_lower = label.lower()
    
    for item, material in MATERIAL_TYPES.items():
        if item in label_lower:
            return material
    
    return "unknown"

def analyze_waste_composition(detected_objects: List[Dict]) -> Dict:
    """Analyze waste composition"""
    composition = {
        "special_waste": 0.0,
        "recyclable": 0.0,
        "residual": 0.0
    }
    
    for obj in detected_objects:
        category = obj.get("category", "Unknown")
        confidence = obj.get("confidence", 0) / 100.0
        
        if category == "Special Waste":
            composition["special_waste"] += confidence
        elif category == "Recyclable":
            composition["recyclable"] += confidence
        elif category == "Residual / Non-Recyclable":
            composition["residual"] += confidence
    
    total = sum(composition.values())
    if total > 0:
        for key in composition:
            composition[key] = round((composition[key] / total) * 100, 2)
    
    return composition

def determine_overall_category(detected_objects: List[Dict]) -> Tuple[str, float]:
    """Determine overall waste category"""
    if not detected_objects:
        return "Unknown", 0.0
    
    category_scores = {"Special Waste": 0.0, "Recyclable": 0.0, 
                      "Residual / Non-Recyclable": 0.0, "Unknown": 0.0}
    
    for obj in detected_objects:
        category = obj.get("category", "Unknown")
        confidence = obj.get("confidence", 0) / 100.0
        category_scores[category] += confidence
    
    # Find category with highest score
    overall_category = max(category_scores, key=category_scores.get)
    total_confidence = sum(category_scores.values())
    
    if total_confidence > 0:
        overall_confidence = round((category_scores[overall_category] / total_confidence) * 100, 2)
    else:
        overall_confidence = 0.0
    
    return overall_category, overall_confidence

def generate_recycling_tips(overall_category: str, detected_objects: List[Dict]) -> List[str]:
    """Generate recycling tips"""
    tips = []
    
    if overall_category == "Recyclable":
        tips = [
            "✅ RECYCLABLE ITEMS DETECTED:",
            "• Rinse cans and bottles before recycling",
            "• Remove caps from plastic bottles",
            "• Flatten cardboard and paper to save space",
            "• Glass bottles can be recycled endlessly",
            "• Check local recycling guidelines for specifics"
        ]
    elif overall_category == "Residual / Non-Recyclable":
        tips = [
            "⚠️ NON-RECYCLABLE ITEMS DETECTED:",
            "• Styrofoam cups are not typically recyclable",
            "• Consider reusable alternatives",
            "• Dispose in regular trash",
            "• Check if your area has special foam recycling",
            "• Reduce use of disposable foam products"
        ]
    elif overall_category == "Special Waste":
        tips = [
            "🚨 SPECIAL WASTE DETECTED:",
            "• Handle with care",
            "• Follow local hazardous waste guidelines",
            "• Do not mix with regular trash",
            "• Use designated collection points",
            "• Wear protective gear when handling"
        ]
    else:
        tips = [
            "ℹ️ WASTE MANAGEMENT TIPS:",
            "• Separate waste properly",
            "• Clean recyclables before disposal",
            "• Reduce, reuse, recycle",
            "• Check local waste management rules",
            "• When in doubt, contact local authorities"
        ]
    
    return tips[:3]  # Return top 3 tips for live detection

def detect_objects_in_image(image_array: np.ndarray, is_live: bool = False, quality: int = 50) -> List[Dict]:
    """Detect objects in image using YOLO"""
    detected_objects = []
    
    if (is_live and live_model is None) or (not is_live and model is None):
        # Debug mode with sample detection
        return [{
            "label": "plastic bottle",
            "confidence": 85.5,
            "box": [0.1, 0.1, 0.4, 0.6],
            "material": "plastic",
            "category": "Recyclable",
            "area_percentage": 25.0,
            "features": {}
        }]
    
    try:
        # Save temp image
        temp_path = f"temp_{int(time.time())}_{np.random.randint(1000)}.jpg"
        
        # Resize for faster processing in live mode
        if is_live:
            height, width = image_array.shape[:2]
            # Reduce size based on quality
            scale_factor = quality / 100.0
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            if new_width > 0 and new_height > 0:
                image_array = cv2.resize(image_array, (new_width, new_height))
        
        cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        # Use appropriate model
        detection_model = live_model if is_live else model
        
        # Adjust confidence for live detection
        conf_threshold = 0.15 if is_live else 0.25
        
        # Run detection
        results = detection_model(temp_path, conf=conf_threshold, iou=0.3)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = detection_model.names[cls]
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    height, width = image_array.shape[:2]
                    
                    # Extract features for ML
                    y1_int, y2_int = int(y1), int(y2)
                    x1_int, x2_int = int(x1), int(x2)
                    
                    if y2_int > y1_int and x2_int > x1_int:
                        image_crop = image_array[y1_int:y2_int, x1_int:x2_int]
                        
                        if is_live:
                            features = learning_system.extract_fast_features(image_crop)
                        else:
                            features = learning_system.extract_features(image_crop, [x1, y1, x2, y2])
                        
                        # Classify using ML if trained
                        if learning_system.is_trained and not is_live:
                            ml_label, ml_confidence = learning_system.predict_category(features)
                            if ml_confidence > 50:  # Use ML prediction if confident
                                label = ml_label
                        
                        # Determine category and material
                        category = classify_object(label)
                        material = get_material_type(label)
                        
                        # Calculate area percentage
                        box_area = (x2 - x1) * (y2 - y1)
                        image_area = width * height
                        area_percentage = (box_area / image_area) * 100 if image_area > 0 else 0
                        
                        detected_objects.append({
                            "label": label,
                            "confidence": round(conf * 100, 1),
                            "box": [round(x1/width, 4), round(y1/height, 4), 
                                   round(x2/width, 4), round(y2/height, 4)],
                            "material": material,
                            "category": category,
                            "area_percentage": round(area_percentage, 1),
                            "features": features
                        })
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"❌ Detection error: {e}")
    
    return detected_objects

@app.post("/detect", response_model=DetectionResponse)
async def detect_waste(request: DetectionRequest, background_tasks: BackgroundTasks):
    """Main waste detection endpoint"""
    start_time = time.time()
    
    try:
        # Process image
        image_base64 = request.image
        if len(image_base64) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large")
        
        # Convert base64 to image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_array = np.array(image)
        
        # Detect objects
        detected_objects = detect_objects_in_image(
            image_array, 
            is_live=request.live_detection,
            quality=request.quality or 50
        )
        
        # Add to dataset if user consented
        if request.allow_training and not request.live_detection:
            background_tasks.add_task(
                learning_system.add_to_dataset,
                detected_objects,
                image_array,
                image.size,
                request.allow_training,
                request.user_id
            )
        
        # Analyze results
        overall_category, overall_confidence = determine_overall_category(detected_objects)
        composition = analyze_waste_composition(detected_objects)
        tips = generate_recycling_tips(overall_category, detected_objects)
        
        # Only upload to Cloudinary for non-live detection
        cloudinary_url = None
        if not request.live_detection:
            try:
                temp_file = f"upload_{int(time.time())}.jpg"
                image.save(temp_file, format='JPEG', quality=85)
                
                upload_result = cloudinary.uploader.upload(
                    temp_file,
                    folder="waste_detection",
                    public_id=f"waste_{int(time.time())}",
                    overwrite=True
                )
                cloudinary_url = upload_result.get('secure_url')
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"⚠️ Cloudinary upload error: {e}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = DetectionResponse(
            success=True,
            detected_objects=detected_objects,
            overall_category=overall_category,
            overall_confidence=overall_confidence,
            waste_composition=WasteComposition(**composition),
            recycling_tips=tips,
            total_objects_detected=len(detected_objects),
            cloudinary_url=cloudinary_url,
            training_consent=request.allow_training,
            dataset_size=len(learning_system.dataset) if learning_system.dataset is not None else 0,
            message=f"Detected {len(detected_objects)} objects",
            processing_time=round(processing_time, 3),
            live_detection=request.live_detection
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-fast")
async def detect_waste_fast(request: DetectionRequest):
    """Fast detection for live camera with minimal processing"""
    start_time = time.time()
    
    try:
        # Process image
        image_base64 = request.image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode only part of image if it's too large
        max_size = 500 * 1024  # 500KB max for live
        if len(image_base64) > max_size:
            # Use lower quality
            image_data = base64.b64decode(image_base64[:max_size])
        else:
            image_data = base64.b64decode(image_base64)
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_array = np.array(image)
        
        # Resize for faster processing
        height, width = image_array.shape[:2]
        if height > 480 or width > 640:
            image_array = cv2.resize(image_array, (640, 480))
        
        # Detect objects with minimal processing
        detected_objects = []
        
        if live_model is not None:
            # Save temp image
            temp_path = f"temp_fast_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            
            # Fast detection with low confidence
            results = live_model(temp_path, conf=0.1, iou=0.2, max_det=10)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = live_model.names[cls]
                        
                        # Fast classification
                        category = classify_object(label)
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        height, width = image_array.shape[:2]
                        
                        detected_objects.append({
                            "label": label,
                            "confidence": round(conf * 100, 1),
                            "box": [round(x1/width, 4), round(y1/height, 4), 
                                   round(x2/width, 4), round(y2/height, 4)],
                            "material": get_material_type(label),
                            "category": category,
                            "area_percentage": round(((x2 - x1) * (y2 - y1)) / (width * height) * 100, 1)
                        })
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "detected_objects": detected_objects,
            "total_objects_detected": len(detected_objects),
            "processing_time": round(processing_time, 3),
            "fps": round(1 / processing_time, 1) if processing_time > 0 else 0
        }
        
    except Exception as e:
        print(f"Fast detection error: {e}")
        return {"success": False, "detected_objects": [], "error": str(e)}

@app.post("/detect-live-batch")
async def detect_live_batch(request: LiveDetectionRequest):
    """Process multiple frames for live detection"""
    start_time = time.time()
    all_detections = []
    
    try:
        # Process only first 3 frames for speed
        frames_to_process = request.frames[:3]
        
        for frame_base64 in frames_to_process:
            try:
                # Decode frame
                if ',' in frame_base64:
                    frame_base64 = frame_base64.split(',')[1]
                
                image_data = base64.b64decode(frame_base64)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image_array = np.array(image)
                
                # Resize for speed
                height, width = image_array.shape[:2]
                if height > 320 or width > 240:
                    image_array = cv2.resize(image_array, (240, 320))
                
                # Quick detection
                temp_path = f"temp_batch_{int(time.time())}_{np.random.randint(1000)}.jpg"
                cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                
                if live_model is not None:
                    results = live_model(temp_path, conf=0.1, iou=0.3, max_det=5)
                    
                    for r in results:
                        boxes = r.boxes
                        if boxes is not None:
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                label = live_model.names[cls]
                                
                                category = classify_object(label)
                                
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                height, width = image_array.shape[:2]
                                
                                all_detections.append({
                                    "label": label,
                                    "confidence": round(conf * 100, 1),
                                    "category": category,
                                    "material": get_material_type(label),
                                    "timestamp": time.time()
                                })
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
        
        # Group detections by label
        grouped_detections = {}
        for detection in all_detections:
            label = detection['label']
            if label not in grouped_detections:
                grouped_detections[label] = {
                    'label': label,
                    'category': detection['category'],
                    'material': detection['material'],
                    'count': 0,
                    'avg_confidence': 0
                }
            grouped_detections[label]['count'] += 1
            grouped_detections[label]['avg_confidence'] = (
                (grouped_detections[label]['avg_confidence'] * (grouped_detections[label]['count'] - 1) + 
                 detection['confidence']) / grouped_detections[label]['count']
            )
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": list(grouped_detections.values()),
            "total_frames_processed": len(frames_to_process),
            "total_detections": len(all_detections),
            "processing_time": round(processing_time, 3),
            "fps": round(len(frames_to_process) / processing_time, 1) if processing_time > 0 else 0
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# WebSocket endpoint for real-time video streaming
@app.websocket("/ws/live-detection")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Process frame
            frame_base64 = frame_data.get('frame', '')
            if frame_base64:
                if ',' in frame_base64:
                    frame_base64 = frame_base64.split(',')[1]
                
                try:
                    # Decode and process
                    image_data = base64.b64decode(frame_base64)
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    image_array = np.array(image)
                    
                    # Resize for speed
                    image_array = cv2.resize(image_array, (320, 240))
                    
                    # Fast detection
                    detections = []
                    if live_model is not None:
                        temp_path = f"temp_ws_{int(time.time())}.jpg"
                        cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                        
                        results = live_model(temp_path, conf=0.1, iou=0.2, max_det=3)
                        
                        for r in results:
                            boxes = r.boxes
                            if boxes is not None:
                                for box in boxes:
                                    cls = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    label = live_model.names[cls]
                                    
                                    detections.append({
                                        'label': label,
                                        'confidence': round(conf * 100, 1),
                                        'category': classify_object(label)
                                    })
                        
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    # Send detections back
                    await websocket.send_text(json.dumps({
                        'success': True,
                        'detections': detections,
                        'timestamp': time.time()
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        'success': False,
                        'error': str(e)
                    }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train the ML classifier"""
    try:
        if len(learning_system.dataset) < request.min_samples:
            return TrainingResponse(
                success=False,
                message=f"Not enough samples. Need {request.min_samples}, have {len(learning_system.dataset)}",
                dataset_size=len(learning_system.dataset)
            )
        
        success, accuracy = learning_system.train_classifier(test_size=request.test_size)
        
        if success:
            return TrainingResponse(
                success=True,
                message=f"Model trained successfully with {accuracy:.2%} accuracy",
                accuracy=accuracy,
                dataset_size=len(learning_system.dataset),
                new_classes=list(learning_system.label_encoder.classes_)
            )
        else:
            return TrainingResponse(
                success=False,
                message="Training failed",
                dataset_size=len(learning_system.dataset)
            )
            
    except Exception as e:
        return TrainingResponse(
            success=False,
            message=f"Error during training: {str(e)}",
            dataset_size=len(learning_system.dataset) if learning_system.dataset is not None else 0
        )

@app.get("/dataset-info", response_model=DatasetInfo)
async def get_dataset_info():
    """Get information about the training dataset"""
    try:
        if learning_system.dataset is None or learning_system.dataset.empty:
            return DatasetInfo(
                total_samples=0,
                classes_count={}
            )
        
        class_counts = learning_system.dataset['label'].value_counts().to_dict()
        
        return DatasetInfo(
            total_samples=len(learning_system.dataset),
            classes_count=class_counts,
            last_trained=learning_system.model_path.stat().st_mtime if learning_system.model_path.exists() else None,
            accuracy=None  # Simplified for now
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live-status")
async def live_status():
    """Check live detection status"""
    return {
        "status": "active",
        "live_model_loaded": live_model is not None,
        "main_model_loaded": model is not None,
        "classifier_trained": learning_system.is_trained,
        "dataset_size": len(learning_system.dataset) if learning_system.dataset is not None else 0,
        "supported_categories": WASTE_CATEGORIES,
        "max_frame_size": "640x480",
        "recommended_fps": 2,
        "endpoints": {
            "POST /detect-fast": "Fast single frame detection",
            "POST /detect-live-batch": "Batch frame processing",
            "WS /ws/live-detection": "WebSocket for real-time streaming"
        }
    }

if __name__ == "__main__":
    print("🚀 Waste Detection API with Live Camera Support")
    print("📊 Initial Dataset Classes: Can, Glass Bottle, Plastic Bottle, Styrofoam Cups, Paper")
    print("🏷️ Waste Categories: Recyclable, Residual/Non-Recyclable, Special Waste")
    print("🎥 Live Detection: Enabled")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"💾 Current dataset size: {len(learning_system.dataset) if learning_system.dataset is not None else 0}")
    print("\n📡 Live Detection Endpoints:")
    print("  • POST /detect-fast - Fast detection for live camera")
    print("  • POST /detect-live-batch - Process multiple frames")
    print("  • WS /ws/live-detection - WebSocket for real-time streaming")
    print("  • GET /live-status - Check live detection status")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )