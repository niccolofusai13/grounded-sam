import os
import sys

# Add GroundingDINO to Python path
grounding_dino_path = "/home/ubuntu/couno/Open-GroundingDino"
sys.path.append(grounding_dino_path)

# Now import GroundingDINO modules
from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

"""
Hyperparam for Ground and Tracking
"""
TEXT_PROMPT = "bowl . plate . cup . chopstick . plastic bottle . packet . spoon . fork . container . foil packet . plastic container . transparent plastic packet . cardboard food container"
PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]
VIDEO_DIR = "notebooks/videos/pi"


# GroundingDINO specific parameters
CONFIG_FILE = "/home/ubuntu/couno/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "/home/ubuntu/couno/Open-GroundingDino/output_final/checkpoint0014.pth"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25


"""
Step 1: Environment settings and model initialization for SAM 2
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


"""
Step X-1: Download frames
"""


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(VIDEO_DIR)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=VIDEO_DIR)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)


"""
Step X: Define Helper functions
"""

# Helper functions for GroundingDINO
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    return boxes_filt, logits_filt

"""
Step 2: Load GroundingDINO model
"""
grounding_dino_model = load_model(CONFIG_FILE, CHECKPOINT_PATH, device)
print(grounding_dino_model)



"""
Step 4: Run GroundingDINO inference
"""
img_path = os.path.join(VIDEO_DIR, frame_names[ann_frame_idx])
print(f"image path is : {img_path}")
image_pil, image = load_image(img_path)
boxes_filt, logits_filt = get_grounding_output(
    grounding_dino_model, image, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, device
)

# Convert boxes to the format expected by SAM2
input_boxes = box_ops.box_cxcywh_to_xyxy(boxes_filt)
input_boxes = input_boxes * torch.Tensor([image_pil.width, image_pil.height, image_pil.width, image_pil.height])
print(f"input boxes are: {input_boxes}")
# Extract confidences and class names
confidences = logits_filt.max(dim=1)[0].tolist()
print(f"confidences ARE: {confidences}")

class_names = [TEXT_PROMPT] * len(boxes_filt)  # Assuming all detections are of the same class
print(f"class_names ARE: {class_names}")

# Create ID_TO_OBJECTS dictionary
OBJECTS = list(set(class_names))  # Get unique class names
ID_TO_OBJECTS = {i+1: obj for i, obj in enumerate(OBJECTS)}
print(f"OBJECTS ARE: {OBJECTS}")
print(f"ID_TO_OBJECTS ARE: {ID_TO_OBJECTS}")

"""
Step 5: Get masks from SAM2
"""

image_predictor.set_image(np.array(image_pil.convert("RGB")))
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes.numpy(),
    multimask_output=False,
)
# convert the mask shape to (n, H, W)
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 6: Register objects with video predictor
"""
if PROMPT_TYPE_FOR_VIDEO == "point":
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
    for object_id, points in enumerate(all_sample_points, start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, box in enumerate(input_boxes, start=1):
        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box.numpy(),
        )
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, mask in enumerate(masks, start=1):
        labels = np.ones((1), dtype=np.int32)
        video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only supports point/box/mask prompts")

"""
Step 7: Propagate video predictor
"""
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
print(f"video segements are: {video_segments}")

"""
Step 8: Visualize and save results
"""
os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks,
        class_id=np.array(object_ids, dtype=np.int32),
    )
    
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

"""
Step 9: Create output video
"""
create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)

print(f"Tracking results saved to {OUTPUT_VIDEO_PATH}")