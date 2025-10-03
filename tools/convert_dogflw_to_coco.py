#!/usr/bin/env python3
# Converts DogFLW's per-image JSONs to COCO keypoints for MMPose (top-down).
# Expects DogFLW laid out like:
#   DOGFLW_ROOT/
#     train/
#       images/*.png
#       labels/*.json   # {"labels": [[x,y],...], "bounding_boxes": [x1,y1,x2,y2]}
#     test/
#       images/*.png
#       labels/*.json
import os, json, glob, argparse
from PIL import Image

def convert_split(split_dir: str, out_json_path: str, category_name='dog_face'):
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    img_paths = sorted(glob.glob(os.path.join(images_dir, '*')))
    images, annotations = [], []
    ann_id = 1
    num_kpts_inferred = None

    for img_id, img_path in enumerate(img_paths, 1):
        name = os.path.basename(img_path)
        lab_path = os.path.join(labels_dir, os.path.splitext(name)[0] + '.json')
        if not os.path.isfile(lab_path):
            continue

        w, h = Image.open(img_path).size
        with open(lab_path, 'r') as f:
            lab = json.load(f)

        kps = lab.get('labels') or lab.get('landmarks')
        if kps is None:
            raise ValueError(f"No 'labels' in {lab_path}")
        if num_kpts_inferred is None:
            num_kpts_inferred = len(kps)

        # bbox could be [x1,y1,x2,y2] or dict; we handle the former (as per README)
        bbox = lab.get('bounding_boxes') or lab.get('bbox')
        if isinstance(bbox, dict):
            x, y, wbox, hbox = float(bbox['x']), float(bbox['y']), float(bbox['width']), float(bbox['height'])
        else:
            # Filter out empty strings and convert to float
            bbox_vals = [float(b) for b in bbox if b and str(b).strip()]
            if len(bbox_vals) != 4:
                print(f"Warning: Invalid bbox in {lab_path}, skipping")
                continue
            x1, y1, x2, y2 = bbox_vals
            x, y, wbox, hbox = x1, y1, (x2 - x1), (y2 - y1)

        keypoints = []
        for (kx, ky) in kps:
            kx, ky = float(kx), float(ky)
            v = 2 if (kx >= 0 and ky >= 0) else 0  # DogFLW annotates occluded points; if absent, mark invisible
            keypoints.extend([kx, ky, v])

        images.append(dict(
            id=img_id, file_name=name, width=w, height=h
        ))
        annotations.append(dict(
            id=ann_id, image_id=img_id, category_id=1,
            keypoints=keypoints, num_keypoints=len(kps),
            bbox=[x, y, wbox, hbox], area=wbox*hbox, iscrowd=0
        ))
        ann_id += 1

    categories = [dict(
        id=1, name=category_name, supercategory='dog',
        keypoints=[f'kp_{i+1}' for i in range(num_kpts_inferred or 46)],
        skeleton=[]
    )]
    coco = dict(images=images, annotations=annotations, categories=categories)
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(coco, f)
    print(f"Wrote {out_json_path}  | images={len(images)} anns={len(annotations)} keypoints={num_kpts_inferred or 46}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dogflw_root', required=True, help='Path containing train/ and test/')
    ap.add_argument('--out_dir', default='data/dogflw/annotations', help='Where to write COCO JSONs')
    args = ap.parse_args()
    convert_split(os.path.join(args.dogflw_root, 'train'), os.path.join(args.out_dir, 'train.json'))
    convert_split(os.path.join(args.dogflw_root, 'test'), os.path.join(args.out_dir, 'val.json'))

