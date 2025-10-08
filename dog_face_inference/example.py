"""
Example usage of Dog Face Inference
"""
from dog_face_predictor import DogFacePredictor
import cv2
import os

def example_single_image():
    """Example: Predict on a single image"""
    
    # Initialize predictor
    predictor = DogFacePredictor(
        config_path='configs/dog_face_config.py',
        checkpoint_path='models/dog_face_model.pth',  # Update this path
        device='cuda:0'  # Use 'cpu' if no GPU
    )
    
    # Run inference
    result = predictor.predict('path/to/dog_image.jpg')
    
    # Access results
    print(f"Detected {result['visible_landmarks']} landmarks")
    print(f"Average confidence: {result['avg_confidence']:.3f}")
    print(f"Extracted {len(result['regions'])} facial regions")
    
    # Save individual regions
    for region_name, region_img in result['regions'].items():
        cv2.imwrite(f'output_{region_name}.jpg', region_img)
        print(f"  Saved {region_name}: {region_img.shape}")


def example_batch_processing():
    """Example: Process multiple images"""
    import glob
    
    predictor = DogFacePredictor(
        config_path='configs/dog_face_config.py',
        checkpoint_path='models/dog_face_model.pth',
        device='cuda:0'
    )
    
    # Get all images
    image_paths = glob.glob('images/*.jpg')
    
    # Process batch
    results = predictor.predict_batch(image_paths)
    
    # Filter high confidence results
    good_results = [r for r in results if r['avg_confidence'] > 0.7]
    
    print(f"Processed {len(results)} images")
    print(f"High confidence: {len(good_results)} images")
    
    return results


def example_with_visualization():
    """Example: Save visualizations"""
    
    predictor = DogFacePredictor(
        config_path='configs/dog_face_config.py',
        checkpoint_path='models/dog_face_model.pth',
        device='cuda:0'
    )
    
    # Save all visualizations
    predictor.save_visualization(
        'path/to/dog_image.jpg',
        'output_folder/',
        save_regions=True,
        save_landmarks=True,
        save_bbox=True
    )
    
    print("✅ Saved visualizations to output_folder/")


def example_region_centers():
    """Example: Get center points of facial regions"""
    
    predictor = DogFacePredictor(
        config_path='configs/dog_face_config.py',
        checkpoint_path='models/dog_face_model.pth',
        device='cuda:0'
    )
    
    result = predictor.predict('path/to/dog_image.jpg')
    
    # Get region centers
    centers = predictor.get_region_centers(result['landmarks'])
    
    for region, (x, y) in centers.items():
        print(f"{region}: center at ({x:.1f}, {y:.1f})")


def example_integration():
    """Example: Clean integration into your project"""
    
    from dog_face_predictor import predict_dog_face
    
    # One-liner prediction
    result = predict_dog_face(
        'dog.jpg',
        'configs/dog_face_config.py',
        'models/dog_face_model.pth'
    )
    
    # Use the results in your pipeline
    if result['avg_confidence'] > 0.7:
        # Process the facial regions
        nose_img = result['regions']['nose']
        eyes_img = [result['regions']['left_eye'], result['regions']['right_eye']]
        
        # Your custom processing here
        # ...
        
        return True
    else:
        print("Low confidence detection, skipping...")
        return False


if __name__ == '__main__':
    print("Dog Face Inference Examples")
    print("="*50)
    
    # Uncomment the example you want to run:
    
    # example_single_image()
    # example_batch_processing()
    # example_with_visualization()
    # example_region_centers()
    # example_integration()
    
    print("\n📝 Update the paths in this file and uncomment an example to run!")
