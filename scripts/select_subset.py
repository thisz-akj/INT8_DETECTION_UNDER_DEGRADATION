import json
import random
from pycocotools.coco import COCO

random.seed(42)

ANN_PATH = "data/annotations/annotations/instances_val2017.json"
TARGET_CLASSES = ["person", "car", "bicycle", "traffic light", "stop sign"]
N_EVAL = 500
N_CALIB = 120

coco = COCO(ANN_PATH)
cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
cats = coco.loadCats(cat_ids)
print("Target categories:", [(c["id"], c["name"]) for c in cats])

img_id_set = set()
for cid in cat_ids:
    img_id_set.update(coco.getImgIds(catIds=[cid]))
img_ids = sorted(img_id_set)
print(f"Images containing >=1 instance of target classes: {len(img_ids)}")

random.shuffle(img_ids)
eval_ids = sorted(img_ids[:N_EVAL])
calib_ids = sorted(img_ids[N_EVAL:N_EVAL + N_CALIB])
print(f"Selected eval set: {len(eval_ids)} images, calib set: {len(calib_ids)} images (disjoint)")

# Build a COCO-format subset json for the eval set, restricted to target categories
images = coco.loadImgs(eval_ids)
ann_ids = coco.getAnnIds(imgIds=eval_ids, catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(ann_ids)

subset = {
    "info": coco.dataset.get("info", {}),
    "licenses": coco.dataset.get("licenses", []),
    "categories": cats,
    "images": images,
    "annotations": anns,
}
with open("data/annotations/instances_val2017_subset500.json", "w") as f:
    json.dump(subset, f)

with open("data/eval_image_ids.txt", "w") as f:
    for i in images:
        f.write(f"{i['id']}\t{i['file_name']}\n")

calib_images = coco.loadImgs(calib_ids)
with open("data/calib_image_ids.txt", "w") as f:
    for i in calib_images:
        f.write(f"{i['id']}\t{i['file_name']}\n")

# quick stats
from collections import Counter
cat_name_by_id = {c["id"]: c["name"] for c in cats}
cnt = Counter(cat_name_by_id[a["category_id"]] for a in anns)
print("Annotation counts per class in eval subset:", dict(cnt))
print(f"Total annotations in eval subset (target classes only): {len(anns)}")
print("Wrote: instances_val2017_subset500.json, eval_image_ids.txt, calib_image_ids.txt")
