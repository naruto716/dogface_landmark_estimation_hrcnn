"""Create a summary image showing all facial regions."""
import cv2
import numpy as np
import os

def create_region_summary():
    """Create a summary showing the updated landmark mappings."""
    # Create a blank canvas
    canvas = np.ones((800, 1200, 3), dtype=np.uint8) * 240
    
    # Title
    cv2.putText(canvas, "Dog Face Landmark Regions (Updated)", (350, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Region information
    regions_info = [
        ("Left Eye", [17, 19, 21, 23], (255, 0, 0)),
        ("Right Eye", [16, 18, 20, 22], (255, 255, 0)),
        ("Nose", [25, 26, 27, 32, 33, 34, 35], (0, 0, 255)),
        ("Mouth", [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45], (0, 165, 255)),
        ("Left Ear", [1, 3, 5, 7, 9, 11, 13], (0, 255, 0)),
        ("Right Ear", [0, 2, 4, 6, 8, 10, 12], (0, 255, 127)),
        ("Forehead", [14, 15, 0, 1, 20, 21], (128, 0, 128))
    ]
    
    y_pos = 120
    for region_name, landmarks, color in regions_info:
        # Draw color box
        cv2.rectangle(canvas, (50, y_pos - 20), (90, y_pos), color, -1)
        cv2.rectangle(canvas, (50, y_pos - 20), (90, y_pos), (0, 0, 0), 1)
        
        # Write region name
        cv2.putText(canvas, region_name + ":", (110, y_pos - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write landmark numbers
        landmarks_str = str(landmarks)
        if len(landmarks_str) > 80:
            # Break long lists into multiple lines
            lines = []
            current_line = ""
            for num in landmarks:
                if len(current_line) > 70:
                    lines.append(current_line.rstrip(", "))
                    current_line = ""
                current_line += f"{num}, "
            if current_line:
                lines.append(current_line.rstrip(", "))
            
            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (300, y_pos - 5 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
            y_pos += 25 * (len(lines) - 1)
        else:
            cv2.putText(canvas, landmarks_str, (300, y_pos - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        y_pos += 50
    
    # Add notes
    cv2.putText(canvas, "Notes:", (50, y_pos + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    notes = [
        "- Some landmarks belong to multiple regions (e.g., 25-27, 32-35 are in both nose and mouth)",
        "- Forehead shares landmarks with ears (0, 1) and eyes (20, 21)",
        "- Total landmarks: 46 (indexed 0-45)",
        "- Bounding boxes use 10% padding by default"
    ]
    
    y_note = y_pos + 60
    for note in notes:
        cv2.putText(canvas, note, (70, y_note),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        y_note += 25
    
    # Save the summary
    cv2.imwrite("corrected_regions_output/regions_summary.png", canvas)
    print("Created summary image: corrected_regions_output/regions_summary.png")

if __name__ == "__main__":
    create_region_summary()
