# Model Checkpoint

Place your trained DogFLW checkpoint here.

## Required File

```
models/
└── dog_face_model.pth  (or any .pth file)
```

## Where to Get the Checkpoint

Copy your trained model from SageMaker:

```bash
# On SageMaker
cd ~/LostPet/dogface_landmark_estimation_hrcnn
cp work_dirs/mmpose_dogflw/best_NME_epoch_100.pth ~/dog_face_model.pth

# Then download to your local machine or copy to your project
```

## File Size

The HRNet-W32 checkpoint is typically ~100-150 MB.

## Verification

Your checkpoint should:
- Be trained on DogFLW dataset (46 keypoints)
- Have `.pth` extension
- Match the config in `configs/dog_face_config.py`
