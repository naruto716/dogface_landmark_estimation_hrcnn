"""
Dog Face Inference Package

Clean, minimal package for dog face landmark detection and facial region extraction.
"""

from .dog_face_predictor import DogFacePredictor, predict_dog_face
from .landmark_regions import DogFacialRegionCropper

__version__ = '1.0.0'
__all__ = ['DogFacePredictor', 'predict_dog_face', 'DogFacialRegionCropper']
