#%%

import torch
import torchvision
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils import show_mask, show_points, show_box
from sam2.build_sam import build_sam2_video_predictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda:3")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
case_no = "4386880"
video_dir = f"./video_images/{case_no}"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

#%%

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

#%%

inference_state = predictor.init_state(video_path=video_dir)

#%%
predictor.reset_state(inference_state)


#%% three objects: one with two clicks and one with a box

ann_frame_idx = 0  # frame you interact with

# ---------- object 1 (id=1): two positive clicks ----------
obj1_id = 1
pts1 = np.array([[320, 60], [340, 30]], dtype=np.float32)       # <-- your coords
lbl1 = np.array([1, 1], dtype=np.int32)
_, oids1, logits1 = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=obj1_id,
    points=pts1, labels=lbl1,
)

# ---------- object 2 (id=2): box ----------
obj2_id = 2
box2 = np.array([450, 200, 700, 420], dtype=np.float32)  # [x1,y1,x2,y2]
_, oids2, logits2 = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=obj2_id,
    box=box2,
)


# ---- collect masks for the interacted frame (all three objects) ----
masks_on_ann = {}
for oids, logits in [(oids1, logits1), (oids2, logits2)]:
    m = (logits > 0.0).cpu().numpy()   # shape: [k, 1, H, W]
    for i, oid in enumerate(oids):
        masks_on_ann[int(oid)] = m[i, 0]  # boolean HxW

# ---- visualize the annotated frame ----
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

# draw prompts
show_points(pts1, lbl1, plt.gca())

show_box(box2, plt.gca())

# draw all masks with stable colors (use obj_id for color/legend)
for oid, mask in sorted(masks_on_ann.items()):
    show_mask(mask, plt.gca(), obj_id=oid)

plt.show()

#%%

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

#%%

show_frame = 1
plt.close("all")
plt.figure(figsize=(6, 4))
plt.title(f"frame {show_frame}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[show_frame])))
for out_obj_id, out_mask in video_segments[show_frame].items():
    show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
plt.show()

#%%

from tqdm import tqdm
save_dir = f"results/{case_no}"

for show_frame in tqdm(range(100)):
    plt.close("all")
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {show_frame}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[show_frame])))
    for out_obj_id, out_mask in video_segments[show_frame].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.savefig(os.path.join(save_dir, f"{show_frame:05d}.png"), bbox_inches='tight', dpi=150)
    plt.close()
    #print(f"saved {show_frame:05d}.png")
    #plt.show()

#%%

for show_frame in tqdm(range(100)):
    plt.close("all")
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {show_frame}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[show_frame])))
    for out_obj_id, out_mask in video_segments[show_frame].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.show()


#%%





#%%











#%%





#%%





#%%