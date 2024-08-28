# dds cloudapi for Grounding DINO 1.5
from dds_cloudapi_sdk import Config, Client, DetectionTask, TextPrompt, DetectionModel, DetectionTarget
import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import json

"""
Step 1: Environment settings and model initialization for SAM 2
"""
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

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "notebooks/videos/pi"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""
# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
image = Image.open(img_path)

# Step 1: initialize the config
token = "acb5ee944d3ff954cb5d2c38d1f5cab8"
config = Config(token)

# Step 2: initialize the client
client = Client(config)

# Step 3: run the task by DetectionTask class
# image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
# if you are processing local image file, upload them to DDS server to get the image url
image_url = client.upload_file(img_path)

task = DetectionTask(
    image_url=image_url,
    prompts=[TextPrompt(text="cup.bowl.plate.chopstick.plastic bottle.spoon.fork.packet.container.trash bin.foil packet.plastic container. transparent plastic packet.cardboard food container")],
    targets=[DetectionTarget.BBox],  # detect bbox
    model=DetectionModel.GDino1_5_Pro,  # detect with GroundingDino-1.5-Pro model
)

client.run_task(task)
result = task.result
objects = result.objects  # the list of detected objects

input_boxes = []
confidences = []
class_names = []

for idx, obj in enumerate(objects):
    input_boxes.append(obj.bbox)
    confidences.append(obj.score)
    class_names.append(obj.category)

input_boxes = np.array(input_boxes)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(np.array(image.convert("RGB")))

# process the detection results
OBJECTS = class_names

ID_TO_OBJECTS = {i+1: obj for i, obj in enumerate(OBJECTS)}

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
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
Step 3: Register each object's positive points to video predictor with separate add_new_points call
"""
PROMPT_TYPE_FOR_VIDEO = "box"  # or "point" or "mask"

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
def get_mask_coordinates(mask):
    return np.column_stack(np.where(mask))

video_segments = {}  # video_segments contains the per-frame segmentation results
mask_coordinates = {}  # To store mask coordinates for each frame and object
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    
    # Extract and store coordinates
    mask_coordinates[out_frame_idx] = {
        out_obj_id: get_mask_coordinates(video_segments[out_frame_idx][out_obj_id])
        for out_obj_id in out_obj_ids
    }

# Save mask coordinates to a file
with open('mask_coordinates.json', 'w') as f:
    json.dump({str(k): {str(k2): v2.tolist() for k2, v2 in v.items()} 
               for k, v in mask_coordinates.items()}, f)

"""
Step 5: Detect object movement and track significant movements
"""
def has_object_moved(coords1, coords2, threshold=5):
    mean1 = np.mean(coords1, axis=0)
    mean2 = np.mean(coords2, axis=0)
    distance = np.linalg.norm(mean1 - mean2)
    return distance > threshold

# Modify the movement detection loop
moved_objects = {}
significant_movements = {}
movement_buffer = {}

for frame_idx in range(1, len(mask_coordinates)):
    moved_objects[frame_idx] = []
    for obj_id in mask_coordinates[frame_idx]:
        if obj_id in mask_coordinates[frame_idx - 1]:
            if has_object_moved(mask_coordinates[frame_idx - 1][obj_id],
                                mask_coordinates[frame_idx][obj_id]):
                moved_objects[frame_idx].append(obj_id)
                
                if obj_id not in movement_buffer:
                    movement_buffer[obj_id] = {'start': frame_idx, 'consecutive_frames': 1}
                else:
                    if frame_idx == movement_buffer[obj_id]['start'] + movement_buffer[obj_id]['consecutive_frames']:
                        movement_buffer[obj_id]['consecutive_frames'] += 1
                    else:
                        movement_buffer[obj_id] = {'start': frame_idx, 'consecutive_frames': 1}
                
                if movement_buffer[obj_id]['consecutive_frames'] >= 5:
                    significant_movements[obj_id] = movement_buffer[obj_id]
            else:
                if obj_id in movement_buffer:
                    del movement_buffer[obj_id]

# Prepare the data for the new file
object_movement_data = []
for obj_id, movement in significant_movements.items():
    object_name = ID_TO_OBJECTS[obj_id]
    start_frame = movement['start']
    end_frame = start_frame + movement['consecutive_frames'] - 1
    object_movement_data.append({
        'object_name': object_name,
        'object_id': obj_id,
        'start_frame': start_frame,
        'end_frame': end_frame
    })

# Save the results
with open('significant_object_movements.json', 'w') as f:
    json.dump(object_movement_data, f)


# Save the results
with open('moved_objects.json', 'w') as f:
    json.dump(moved_objects, f)

"""
Step 6: Visualize the segment results across the video and save them
"""
save_dir = "./tracking_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
    
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

"""
Step 7: Convert the annotated frames to video
"""
output_video_path = "./object_tracking_with_movement.mp4"
create_video_from_images(save_dir, output_video_path)

print("Processing complete. Check 'mask_coordinates.json' for mask coordinates and 'moved_objects.json' for detected movements.")