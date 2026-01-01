"""
Django views for Skin Disease Detection web application.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import io
import random
import shutil
import uuid

import cv2
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import joblib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from training.utils import get_project_root, allowed_image_file, compute_severity_level
from .disease_info import get_disease_info


PROJECT_ROOT = get_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Media directories
MEDIA_ROOT = Path(settings.MEDIA_ROOT)
MEDIA_ROOT.mkdir(exist_ok=True)
UPLOADED_IMAGES_DIR = MEDIA_ROOT / "uploaded_images"
UPLOADED_IMAGES_DIR.mkdir(exist_ok=True)


def load_class_names() -> Dict[int, str]:
    """
    Infer class names from the SkinDisease training folders.
    Falls back to hardcoded class names from disease_info if dataset is missing.
    """
    skin_root = PROJECT_ROOT / "SkinDisease" / "SkinDisease" / "train"
    if skin_root.is_dir():
        classes = sorted([d.name for d in skin_root.iterdir() if d.is_dir()])
        if classes:
            idx_to_class = {i: c for i, c in enumerate(classes)}
            return idx_to_class
    
    # Fallback: Use class names from disease_info.py
    from .disease_info import DISEASE_INFO
    classes = sorted(DISEASE_INFO.keys())
    idx_to_class = {i: c for i, c in enumerate(classes)}
    return idx_to_class


IDX_TO_CLASS = load_class_names()


def get_sample_images_from_dataset(disease_name: str, max_images: int = 3) -> list[str]:
    """
    Get up to max_images sample images from the dataset for the given disease class.
    Tries to find skin-related images first (prefers images that look like skin conditions).
    Returns a list of relative URL paths to the images.
    """
    dataset_root = PROJECT_ROOT / "SkinDisease" / "SkinDisease" / "train"
    disease_dir = dataset_root / disease_name
    
    if not disease_dir.exists() or not disease_dir.is_dir():
        return []
    
    # Get all image files
    image_files = list(disease_dir.glob("*.jpeg")) + list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.png"))
    
    if not image_files:
        return []
    
    # Try to prefer skin images over oral/other body parts
    # Filter out images that might be oral (tongue, mouth) - simple heuristic based on common patterns
    skin_images = []
    other_images = []
    
    for img_file in image_files:
        # Simple heuristic: if filename contains certain keywords, it might be oral/other
        filename_lower = img_file.name.lower()
        # Skip obvious oral images (tongue, mouth, oral)
        if any(keyword in filename_lower for keyword in ['tongue', 'oral', 'mouth', 'lip']):
            other_images.append(img_file)
        else:
            skin_images.append(img_file)
    
    # Prefer skin images, fallback to any image
    if skin_images:
        available_images = skin_images
    elif other_images:
        available_images = other_images
    else:
        available_images = image_files
    
    # Shuffle and take up to max_images
    random.shuffle(available_images)
    selected_images = available_images[:max_images]
    
    # Copy to media folder for serving and collect URLs
    media_sample_dir = MEDIA_ROOT / "dataset_samples" / disease_name
    media_sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_urls = []
    for sample_image in selected_images:
        # Use a consistent filename based on the original
        media_sample_path = media_sample_dir / sample_image.name
        
        # Copy if not already exists
        if not media_sample_path.exists():
            shutil.copy2(sample_image, media_sample_path)
        
        # Add relative URL
        sample_urls.append(f"{settings.MEDIA_URL}dataset_samples/{disease_name}/{sample_image.name}")
    
    return sample_urls


def build_dl_transform(input_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


class SimpleCNN(nn.Module):
    """Same architecture as in training/dl_train.py, for inference."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_dl_model(model_type: str):
    """
    Load DL model with graceful fallback between ResNet50 (new) and ResNet18 (old).
    This prevents shape-mismatch errors when older weights exist.
    """
    num_classes = len(IDX_TO_CLASS)
    if num_classes == 0:
        raise RuntimeError(
            "Could not infer class names. Make sure SkinDisease dataset is present."
        )

    if model_type == "dl_resnet":
        # Candidate weight paths (new -> old)
        candidate_weights = [
            MODELS_DIR / "resnet50_skindisease.pt",
            PROJECT_ROOT / "best_skindisease_model.pt",
            MODELS_DIR / "resnet18_skindisease.pt",
        ]

        last_error = None

        # Try ResNet101 first
        for weight_path in candidate_weights:
            if not weight_path.exists():
                continue
            try:
                model = models.resnet101(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )
                model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                # Loaded ResNet101 weights successfully
                print(f"Loaded ResNet101 weights from {weight_path}")
                return model
            except Exception:
                # Try ResNet50 if ResNet101 fails
                try:
                    model = models.resnet50(weights=None)
                    in_features = model.fc.in_features
                    model.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(in_features, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.4),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_classes),
                    )
                    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
                    model.to(DEVICE)
                    model.eval()
                    # Loaded ResNet50 weights successfully
                    print(f"Loaded ResNet50 weights from {weight_path}")
                    return model
                except Exception as e:
                    last_error = e
                    # Log failure to load ResNet101/50, will try next candidate
                    print(f"Failed to load ResNet101/50 from {weight_path}: {e}")
                # Try next candidate (possibly ResNet18 below)

        # Fallback to ResNet18 if ResNet50 load failed
        for weight_path in candidate_weights:
            if not weight_path.exists():
                continue
            try:
                model = models.resnet18(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, num_classes),
                )
                model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                # Loaded ResNet18 weights successfully
                print(f"Loaded ResNet18 weights from {weight_path}")
                return model
            except Exception as e:
                last_error = e
                # Log failure to load ResNet18 and continue to next candidate
                print(f"Failed to load ResNet18 from {weight_path}: {e}")

        # If all attempts failed
        raise RuntimeError(
            "Could not load any DL model weights. "
            f"Last error: {last_error}"
        )

    else:
        # default: simple cnn
        weight_path = MODELS_DIR / "simple_cnn_skindisease.pt"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. "
                "Please run training/dl_train.py or train_skindisease.py first."
            )
        model = SimpleCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model


def load_ml_model(model_name: str):
    """
    Load a classical ML model (SVM / RandomForest / LogisticRegression)
    trained in training/ml_train.py.
    """
    path = MODELS_DIR / f"{model_name}_skin_ml.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"ML model file not found at {path}. "
            "Please run training/ml_train.py first."
        )
    return joblib.load(path)


DL_TRANSFORM = build_dl_transform()


def predict_with_dl(image: Image.Image, model_type: str, use_tta: bool = True) -> Dict[str, Any]:
    """
    Run prediction using a deep learning model with Test-Time Augmentation (TTA) for better accuracy.
    TTA averages predictions from multiple augmented versions of the image.
    """
    model = load_dl_model(model_type)
    
    # Test-Time Augmentation: create multiple augmented versions
    if use_tta:
        augmentations = [
            # Original
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Horizontal Flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Vertical Flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Rotation 10 degrees
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation((10, 10)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Rotation -10 degrees
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation((-10, -10)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        ]
        
        all_probs = []
        with torch.no_grad():
            for aug_transform in augmentations:
                img_tensor = aug_transform(image).unsqueeze(0).to(DEVICE)
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                all_probs.append(probs)
        
        # Average probabilities from all augmentations
        probs = np.mean(all_probs, axis=0)
    else:
        img_tensor = DL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    class_name = IDX_TO_CLASS.get(pred_idx, f"class_{pred_idx}")
    severity = compute_severity_level(confidence)
    
    # Get top 3 predictions sorted by confidence
    top_indices = np.argsort(probs)[::-1][:3]
    top_predictions = [
        {
            "disease": IDX_TO_CLASS.get(int(idx), f"class_{idx}"),
            "confidence": float(probs[int(idx)]),
        }
        for idx in top_indices
    ]

    # Sort probabilities dictionary by value (descending)
    prob_dict = {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

    # Check prediction reliability
    # Low confidence threshold - if below this, prediction is unreliable
    LOW_CONFIDENCE_THRESHOLD = 0.35
    
    # Check if top 2 predictions are close (uncertain prediction)
    second_confidence = float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.0
    confidence_gap = confidence - second_confidence
    
    # Determine if prediction is reliable
    is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    is_uncertain = confidence_gap < 0.15  # Top 2 predictions are close
    is_unreliable = is_low_confidence or is_uncertain
    
    # Special handling for Unknown_Normal
    is_normal_skin = class_name == "Unknown_Normal"

    # Get disease information
    disease_info = get_disease_info(class_name)

    return {
        "model_type": model_type,
        "is_dl_model": model_type.startswith("dl_"),
        "disease_name": class_name,
        "confidence": confidence,
        "severity": severity,
        "top_predictions": top_predictions,  # Top 3 predictions
        "probabilities": sorted_probs,  # All probabilities sorted
        "disease_info": disease_info,  # Disease information, resources, treatment
        "is_low_confidence": is_low_confidence,  # Flag for low confidence
        "is_uncertain": is_uncertain,  # Flag for uncertain prediction
        "is_unreliable": is_unreliable,  # Flag for unreliable prediction
        "is_normal_skin": is_normal_skin,  # Flag for normal/healthy skin
        "confidence_gap": confidence_gap,  # Gap between top 2 predictions
    }


def predict_with_ml(image: Image.Image, model_type: str) -> Dict[str, Any]:
    """
    Run prediction using a classical ML model.
    We reuse the same handcrafted features (color histogram) as in ml_train.py.
    """
    # convert PIL image to BGR numpy via temporary buffer
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    np_img = np.frombuffer(buf.read(), dtype=np.uint8)
    bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Could not decode uploaded image for ML model.")

    from training.utils import extract_color_histogram

    feats = extract_color_histogram(bgr).reshape(1, -1)

    model_key = {
        "ml_svm": "svm",
        "ml_rf": "random_forest",
        "ml_logreg": "log_reg",
    }.get(model_type, "random_forest")

    model = load_ml_model(model_key)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]
        pred_idx = int(np.argmax(probs))
        class_name = model.classes_[pred_idx]
        confidence = float(probs[pred_idx])
    else:
        pred_label = model.predict(feats)[0]
        class_name = str(pred_label)
        confidence = 0.5  # fallback when probability not available

    severity = compute_severity_level(confidence)

    prob_map = {}
    if probs is not None:
        for cls, p in zip(model.classes_, probs):
            prob_map[str(cls)] = float(p)
        
        # Get top 3 predictions sorted by confidence
        sorted_probs = dict(sorted(prob_map.items(), key=lambda x: x[1], reverse=True))
        top_3_items = list(sorted_probs.items())[:3]
        top_predictions = [
            {
                "disease": disease,
                "confidence": prob,
            }
            for disease, prob in top_3_items
        ]
    else:
        # Fallback when probabilities not available
        top_predictions = [
            {
                "disease": class_name,
                "confidence": confidence,
            }
        ]
        prob_map = {class_name: confidence}

    # Sort probabilities dictionary by value (descending)
    sorted_probs = dict(sorted(prob_map.items(), key=lambda x: x[1], reverse=True))

    # Check prediction reliability
    LOW_CONFIDENCE_THRESHOLD = 0.35
    second_confidence = sorted_probs[list(sorted_probs.keys())[1]] if len(sorted_probs) > 1 else 0.0
    confidence_gap = confidence - second_confidence
    
    is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    is_uncertain = confidence_gap < 0.15
    is_unreliable = is_low_confidence or is_uncertain
    is_normal_skin = class_name == "Unknown_Normal"

    # Get disease information
    disease_info = get_disease_info(class_name)

    return {
        "model_type": model_type,
        "is_dl_model": model_type.startswith("dl_"),
        "disease_name": class_name,
        "confidence": confidence,
        "severity": severity,
        "top_predictions": top_predictions,  # Top 3 predictions
        "probabilities": sorted_probs,  # All probabilities sorted
        "disease_info": disease_info,  # Disease information, resources, treatment
        "is_low_confidence": is_low_confidence,
        "is_uncertain": is_uncertain,
        "is_unreliable": is_unreliable,
        "is_normal_skin": is_normal_skin,
        "confidence_gap": confidence_gap,
    }


def index(request):
    """Home page view."""
    return render(request, 'index.html')


def about(request):
    """About page view."""
    return render(request, 'about.html')


@require_http_methods(["GET", "POST"])
def upload(request):
    """Upload and predict view."""
    if request.method == "POST":
        file = request.FILES.get("image")
        model_type = request.POST.get("model_type", "dl_resnet")

        if not file or file.name == "":
            messages.error(request, "Please select an image file.")
            return redirect('skindetection:upload')

        if not allowed_image_file(file.name):
            messages.error(request, "Unsupported file type. Please upload JPG or PNG images.")
            return redirect('skindetection:upload')

        try:
            image = Image.open(file).convert("RGB")
        except Exception:
            messages.error(request, "Could not read the uploaded image.")
            return redirect('skindetection:upload')

        # Save uploaded image to media folder
        file_extension = Path(file.name).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        uploaded_image_path = UPLOADED_IMAGES_DIR / unique_filename
        
        # Save the image
        image.save(uploaded_image_path)
        uploaded_image_url = f"{settings.MEDIA_URL}uploaded_images/{unique_filename}"

        try:
            if model_type.startswith("dl_"):
                result = predict_with_dl(image, model_type=model_type)
            else:
                # ML models may not be trained yet, fallback to DL
                try:
                    result = predict_with_ml(image, model_type=model_type)
                except FileNotFoundError:
                    messages.info(request, "ML model not found. Using Deep Learning model instead.")
                    result = predict_with_dl(image, model_type="dl_resnet")
        except Exception as e:
            messages.error(request, f"Prediction failed: {e}")
            return redirect('skindetection:upload')

        # Get sample images from dataset (up to 3)
        sample_image_urls = get_sample_images_from_dataset(result["disease_name"], max_images=3)
        
        # Add image URLs to result
        result["uploaded_image_url"] = uploaded_image_url
        result["sample_image_urls"] = sample_image_urls  # List of up to 3 images

        return render(request, 'result.html', {'result': result})

    return render(request, 'upload.html')


from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def api_signup(request):
    """API endpoint for user registration."""
    username = request.data.get('username')
    password = request.data.get('password')
    email = request.data.get('email', '')
    
    if not username or not password:
        return Response({"error": "Username and password are required"}, status=status.HTTP_400_BAD_REQUEST)
    
    if User.objects.filter(username=username).exists():
        return Response({"error": "Username already exists"}, status=status.HTTP_400_BAD_REQUEST)
    
    user = User.objects.create_user(username=username, password=password, email=email)
    token, _ = Token.objects.get_or_create(user=user)
    
    return Response({
        "token": token.key,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        }
    }, status=status.HTTP_201_CREATED)

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def api_login(request):
    """API endpoint for user login."""
    username = request.data.get('username')
    password = request.data.get('password')
    
    user = authenticate(username=username, password=password)
    if not user:
        return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
    
    token, _ = Token.objects.get_or_create(user=user)
    return Response({
        "token": token.key,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        }
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_profile(request):
    """API endpoint to get user profile."""
    user = request.user
    return Response({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "date_joined": user.date_joined
    })

@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """REST API endpoint for predictions."""
    file = request.FILES.get("image")
    model_type = request.POST.get("model_type", "dl_resnet")
    
    if file is None or file.name == "":
        return JsonResponse({"error": "No image uploaded"}, status=400)
    
    if not allowed_image_file(file.name):
        return JsonResponse({"error": "Unsupported file type"}, status=400)
    
    try:
        image = Image.open(file).convert("RGB")
        if model_type.startswith("dl_"):
            result = predict_with_dl(image, model_type=model_type)
        else:
            # ML models may not be trained yet, fallback to DL
            try:
                result = predict_with_ml(image, model_type=model_type)
            except FileNotFoundError:
                result = predict_with_dl(image, model_type="dl_resnet")
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

