"""Interactive tool to map landmarks to facial regions using tkinter."""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import json
from PIL import Image, ImageTk, ImageDraw
import os
from config import TRAIN_IMAGES, TRAIN_ANN
import math


class LandmarkMapper:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog Face Landmark Mapper")
        
        # Current state
        self.current_image_idx = 0
        self.images_data = []
        self.current_region = 'left_eye'
        self.landmark_regions = {
            'left_eye': [],
            'right_eye': [],
            'nose': [],
            'mouth': [],
            'left_ear': [],
            'right_ear': [],
            'face_contour': []
        }
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        
        # Colors for each region
        self.region_colors = {
            'left_eye': '#0000FF',      # Blue
            'right_eye': '#00FFFF',     # Cyan
            'nose': '#FF0000',          # Red
            'mouth': '#FFA500',         # Orange
            'left_ear': '#00FF00',      # Green
            'right_ear': '#7FFF00',     # Light green
            'face_contour': '#FF00FF'   # Purple
        }
        
        # Load data
        self.load_data()
        
        # Create UI
        self.create_ui()
        
        # Display first image
        self.display_image()
    
    def load_data(self):
        """Load COCO annotations and prepare image data."""
        with open(TRAIN_ANN, 'r') as f:
            coco_data = json.load(f)
        
        # Create mappings
        img_to_ann = {ann['image_id']: ann for ann in coco_data['annotations']}
        id_to_img = {img['id']: img for img in coco_data['images']}
        
        # Load first 5 images
        count = 0
        for img_id, img_info in id_to_img.items():
            if count >= 5:
                break
            
            if img_id in img_to_ann:
                img_path = os.path.join(TRAIN_IMAGES, img_info['file_name'])
                if os.path.exists(img_path):
                    ann = img_to_ann[img_id]
                    
                    # Convert keypoints
                    keypoints_flat = ann['keypoints']
                    num_keypoints = len(keypoints_flat) // 3
                    
                    keypoints = []
                    for i in range(num_keypoints):
                        x = keypoints_flat[i * 3]
                        y = keypoints_flat[i * 3 + 1]
                        v = keypoints_flat[i * 3 + 2]
                        conf = 1.0 if v > 0 else 0.0
                        keypoints.append([x, y, conf])
                    
                    self.images_data.append({
                        'path': img_path,
                        'filename': img_info['file_name'],
                        'keypoints': np.array(keypoints)
                    })
                    count += 1
    
    def create_ui(self):
        """Create the user interface."""
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Region selection
        ttk.Label(control_frame, text="Current Region:").grid(row=0, column=0, padx=5)
        self.region_var = tk.StringVar(value=self.current_region)
        region_menu = ttk.Combobox(control_frame, textvariable=self.region_var, 
                                  values=list(self.landmark_regions.keys()),
                                  state="readonly", width=15)
        region_menu.grid(row=0, column=1, padx=5)
        region_menu.bind('<<ComboboxSelected>>', self.on_region_change)
        
        # Image navigation
        ttk.Button(control_frame, text="Previous Image", 
                  command=self.prev_image).grid(row=0, column=2, padx=5)
        self.image_label = ttk.Label(control_frame, text="Image 1/5")
        self.image_label.grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Next Image", 
                  command=self.next_image).grid(row=0, column=4, padx=5)
        
        # Instructions
        instructions = ttk.Label(control_frame, 
                               text="Click landmarks to add/remove from selected region | Shift+drag to pan | Scroll to zoom",
                               font=('Arial', 10))
        instructions.grid(row=1, column=0, columnspan=6, pady=10)
        
        # Canvas for image
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg='gray')
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/Linux
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # macOS scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # macOS scroll down
        
        # Zoom controls
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.grid(row=0, column=5, padx=20)
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        # Right panel for current mappings
        mapping_frame = ttk.Frame(self.root, padding="10")
        mapping_frame.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        ttk.Label(mapping_frame, text="Current Mappings:", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5)
        
        self.mapping_text = tk.Text(mapping_frame, width=30, height=30)
        self.mapping_text.grid(row=1, column=0)
        
        # Bottom buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.grid(row=2, column=0, columnspan=2)
        
        ttk.Button(button_frame, text="Clear Current Region", 
                  command=self.clear_current_region).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_all).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Mapping", 
                  command=self.save_mapping).grid(row=0, column=2, padx=5)
        
        # Add note about multi-region support
        note_label = ttk.Label(button_frame, 
                              text="Note: Landmarks can belong to multiple regions (e.g., nose-mouth boundary)",
                              font=('Arial', 9), foreground='gray')
        note_label.grid(row=1, column=0, columnspan=3, pady=5)
    
    def display_image(self):
        """Display current image with landmarks."""
        if not self.images_data:
            return
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Load and resize image
        img_data = self.images_data[self.current_image_idx]
        img = cv2.imread(img_data['path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate display dimensions with zoom
        h, w = img_rgb.shape[:2]
        base_scale = min(800/w, 600/h)
        scale = base_scale * self.zoom_level
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        # Display image with pan offset
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo)
        
        # Store scale for coordinate conversion
        self.scale = scale
        self.base_scale = base_scale
        
        # Draw landmarks
        self.landmark_items = {}
        keypoints = img_data['keypoints']
        
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf > 0:
                # Scale and pan coordinates
                x_scaled = x * self.scale + self.pan_x
                y_scaled = y * self.scale + self.pan_y
                
                # Determine which regions this landmark belongs to
                regions_for_landmark = []
                for region, indices in self.landmark_regions.items():
                    if idx in indices:
                        regions_for_landmark.append(region)
                
                # Draw landmark with multi-region support
                if len(regions_for_landmark) == 0:
                    # Unassigned - white circle
                    circle = self.canvas.create_oval(
                        x_scaled - 8, y_scaled - 8,
                        x_scaled + 8, y_scaled + 8,
                        fill='white', outline='black', width=2,
                        tags=f"landmark_{idx}"
                    )
                elif len(regions_for_landmark) == 1:
                    # Single region - solid color
                    color = self.region_colors[regions_for_landmark[0]]
                    circle = self.canvas.create_oval(
                        x_scaled - 8, y_scaled - 8,
                        x_scaled + 8, y_scaled + 8,
                        fill=color, outline='black', width=2,
                        tags=f"landmark_{idx}"
                    )
                else:
                    # Multiple regions - draw pie slices
                    self.draw_multi_region_landmark(x_scaled, y_scaled, idx, regions_for_landmark)
                    circle = None  # Already drawn
                
                # Draw number with background for visibility
                bg_rect = self.canvas.create_rectangle(
                    x_scaled - 8, y_scaled - 8,
                    x_scaled + 8, y_scaled + 8,
                    fill='', outline='',
                    tags=f"landmark_{idx}"
                )
                text = self.canvas.create_text(
                    x_scaled, y_scaled,
                    text=str(idx), fill='black', font=('Arial', 10, 'bold'),
                    tags=f"landmark_{idx}"
                )
                
                self.landmark_items[idx] = (circle, text)
        
        # Update labels
        self.image_label.config(text=f"Image {self.current_image_idx + 1}/{len(self.images_data)}")
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        
        # Update mappings display
        self.update_mapping_display()
    
    def draw_multi_region_landmark(self, x, y, idx, regions):
        """Draw a landmark that belongs to multiple regions as pie slices."""
        radius = 8
        num_regions = len(regions)
        angle_per_region = 360 / num_regions
        
        for i, region in enumerate(regions):
            start_angle = i * angle_per_region
            color = self.region_colors[region]
            
            # Draw pie slice
            self.canvas.create_arc(
                x - radius, y - radius,
                x + radius, y + radius,
                start=start_angle, extent=angle_per_region,
                fill=color, outline='black', width=2,
                tags=f"landmark_{idx}"
            )
    
    def on_canvas_click(self, event):
        """Handle click on canvas to select/deselect landmarks."""
        # Check if Shift is held for panning
        if event.state & 0x1:  # Shift key
            self.drag_start = (event.x, event.y)
            return
            
        # Find closest landmark
        closest_idx = None
        min_dist = float('inf')
        
        img_data = self.images_data[self.current_image_idx]
        keypoints = img_data['keypoints']
        
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf > 0:
                # Account for zoom and pan
                x_scaled = x * self.scale + self.pan_x
                y_scaled = y * self.scale + self.pan_y
                
                dist = ((event.x - x_scaled) ** 2 + (event.y - y_scaled) ** 2) ** 0.5
                if dist < 20 and dist < min_dist:  # Increased threshold for easier selection
                    min_dist = dist
                    closest_idx = idx
        
        if closest_idx is not None:
            self.toggle_landmark(closest_idx)
    
    def toggle_landmark(self, idx):
        """Toggle landmark assignment to current region (allows multi-region assignment)."""
        current_region = self.region_var.get()
        
        # Toggle membership in current region
        if idx in self.landmark_regions[current_region]:
            # Remove from current region
            self.landmark_regions[current_region].remove(idx)
        else:
            # Add to current region
            self.landmark_regions[current_region].append(idx)
            self.landmark_regions[current_region].sort()
        
        # Redraw
        self.display_image()
    
    def on_region_change(self, event):
        """Handle region selection change."""
        self.current_region = self.region_var.get()
    
    def on_drag(self, event):
        """Handle mouse drag for panning."""
        if self.drag_start and event.state & 0x1:  # Shift key held
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start = (event.x, event.y)
            self.display_image()
    
    def on_drag_release(self, event):
        """Handle mouse release."""
        self.drag_start = None
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        # Get mouse position relative to canvas
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction
        if event.delta > 0 or event.num == 4:  # Scroll up
            self.zoom_level *= 1.1
        else:  # Scroll down
            self.zoom_level *= 0.9
        
        # Limit zoom level
        self.zoom_level = max(0.5, min(5.0, self.zoom_level))
        
        # Redraw
        self.display_image()
    
    def zoom_in(self):
        """Zoom in by 20%."""
        self.zoom_level = min(5.0, self.zoom_level * 1.2)
        self.display_image()
    
    def zoom_out(self):
        """Zoom out by 20%."""
        self.zoom_level = max(0.5, self.zoom_level * 0.8)
        self.display_image()
    
    def reset_view(self):
        """Reset zoom and pan."""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.display_image()
    
    def prev_image(self):
        """Go to previous image."""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.display_image()
    
    def next_image(self):
        """Go to next image."""
        if self.current_image_idx < len(self.images_data) - 1:
            self.current_image_idx += 1
            self.display_image()
    
    def clear_current_region(self):
        """Clear landmarks from current region."""
        current_region = self.region_var.get()
        self.landmark_regions[current_region] = []
        self.display_image()
    
    def clear_all(self):
        """Clear all mappings."""
        for region in self.landmark_regions:
            self.landmark_regions[region] = []
        self.display_image()
    
    def update_mapping_display(self):
        """Update the mapping text display."""
        self.mapping_text.delete(1.0, tk.END)
        
        for region, indices in self.landmark_regions.items():
            color = self.region_colors[region]
            self.mapping_text.insert(tk.END, f"{region}:\n", 'bold')
            
            if indices:
                self.mapping_text.insert(tk.END, f"  {indices}\n", region)
            else:
                self.mapping_text.insert(tk.END, "  (none)\n", 'gray')
            
            self.mapping_text.insert(tk.END, "\n")
        
        # Configure tags for colors
        for region, color in self.region_colors.items():
            self.mapping_text.tag_config(region, foreground=color)
        self.mapping_text.tag_config('bold', font=('Arial', 10, 'bold'))
        self.mapping_text.tag_config('gray', foreground='gray')
    
    def save_mapping(self):
        """Save the landmark mapping to file."""
        # Save to JSON
        output_file = 'landmark_regions_mapping.json'
        with open(output_file, 'w') as f:
            json.dump(self.landmark_regions, f, indent=2)
        
        # Also generate Python code
        python_file = 'landmark_regions_update.py'
        with open(python_file, 'w') as f:
            f.write("# Updated landmark regions mapping\n\n")
            f.write("landmark_regions = {\n")
            for region, indices in self.landmark_regions.items():
                f.write(f"    '{region}': {indices},\n")
            f.write("}\n")
        
        messagebox.showinfo("Success", 
                          f"Mapping saved to:\n- {output_file}\n- {python_file}")


def main():
    root = tk.Tk()
    app = LandmarkMapper(root)
    root.mainloop()


if __name__ == '__main__':
    main()
