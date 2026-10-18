# Data sources and licences — starter model `models/starter_v1`

The starter model is trained on **non-final data**: COCO photographs of full-size people and MuJoCo renders of
the 1:14 scene. It exists so the pipeline (dataset → train → ONNX → wrapper → ROS 2 node) runs end-to-end
before the forklift camera is mounted. It is **not** the PC-08 result; the final model is trained on ≥ 400
images per class from the forklift camera (plan.md B6, [docs/16-vision.md](../../docs/16-vision.md)).

Nothing in `datasets/` is committed (gitignored); every step below regenerates it.

## 1. COCO 2017 — `person`

| | |
|---|---|
| What | `person` boxes from COCO 2017 (Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014) |
| Annotations licence | CC BY 4.0, © COCO Consortium (cocodataset.org/#termsofuse) |
| Image licences | each image keeps its Flickr licence (listed per image in the COCO json) |
| Filter | only licence **4 CC BY 2.0**, **5 CC BY-SA 2.0**, **7 no known copyright restrictions**, **8 U.S. Government work** — every NonCommercial (1–3) and NoDerivs (6) image is excluded, because we crop and composite |
| Also skipped | images with a crowd region (`iscrowd`, unlabelled people) and images with > 8 people |
| Downloaded | `annotations_trainval2017.zip` (241 MB) + the selected images one by one from `images.cocodataset.org`, no login |
| Script | `scripts/fetch_coco_persons.py --direct 700 --cutout-images 400` (seed 0) |
| Attribution list | [`data_manifests/coco_person_manifest.csv`](data_manifests/coco_person_manifest.csv) — COCO id, licence, Flickr URL of every image used (CC BY / BY-SA require attribution; BY-SA derivatives stay BY-SA) |

Two disjoint image sets:

| Set | Source split | Images | Use | Licences |
|---|---|---|---|---|
| direct | val2017 | 587 (all val2017 images that pass the filter) | whole images, 1404 person boxes, class 0 | 401 BY, 182 BY-SA, 4 no-known-restrictions |
| cutout | train2017 | 119 | 144 persons cut out with their COCO polygon mask (upright, ≥ 120 px, not truncated, not overlapping another person), pasted into renders | 72 BY, 44 BY-SA, 3 no-known-restrictions |

## 2. MuJoCo synthetic renders — `pallet`, `forklift`, `obstacle` (+ pasted persons)

| | |
|---|---|
| Generator | `scripts/make_synthetic.py --n 1800 --person-prob 0.35` (seed 14), own code, same licence as this repo |
| Geometry | 1:14 forklift and pallet dimensions read from `code/training/forklift_rl/config.yaml` (PLACEHOLDERS until ME-01); pallet with load box and a DICT_4X4_50 id 7 marker; racks, floor tape, walls as unlabelled distractors; obstacles = boxes, cans, balls, traffic cones, piles of small boxes |
| Randomised per frame | camera height 0.03–0.6 m (70 % at fork-carriage height ≤ 0.22 m), pitch 2–45° down, vertical FOV 38–56°, object poses/colours, light level dim / normal / bright (PC-11; 501 / 778 / 521 frames), motion blur, sensor noise, JPEG quality 60–95 |
| Per scene (60 frames) | floor texture and colours, skybox, pallet scale ×0.85–1.25, obstacle shapes |
| Labels | from MuJoCo's segmentation render (separate context without MSAA, floor reflections off): tight box of the **visible** pixels of each object, ≥ 30 px and ≥ 4 px per side; the pallet box includes its load |
| Persons | in 601 of 1800 frames 1–3 COCO cut-outs (set "cutout" above) are pasted at 70–560 px height, brightness matched to the light level; their pixels overwrite the instance map, so occluded boxes shrink correctly |
| Empty frames | 34 frames have no labelled object (kept as background images) |

## 3. Pretrained weights and software

| | Licence |
|---|---|
| `yolo11n.pt` (Ultralytics YOLO11 nano, COCO-pretrained), downloaded by Ultralytics from its GitHub release v8.4.0 | **AGPL-3.0** |
| Ultralytics 8.4.182 (training, export only) | **AGPL-3.0** |
| PyTorch 2.14.1+cu126, MuJoCo 3.16, OpenCV, onnxruntime, ONNX | BSD / Apache-2.0 / MIT style |

AGPL-3.0 note: weights fine-tuned from `yolo11n.pt` with Ultralytics are treated by Ultralytics as AGPL-3.0
works. For a university project that publishes its source this is fine; a closed commercial product would need
an Ultralytics Enterprise licence (or YOLO weights trained from scratch with another trainer). The runtime
wrapper `forklift_vision/detector.py` does not import Ultralytics.

## 4. Built dataset `datasets/starter_v1`

`scripts/split_dataset.py --out datasets/starter_v1 --chunk 20 datasets/raw/synth_mujoco datasets/raw/coco_person`
(seed 0; `--chunk 20` keeps blocks of 20 consecutive renders — which share a scene — in one split).

| split | images | person | pallet | forklift | obstacle |
|---|---|---|---|---|---|
| train | 1694 (71.0 %) | 849 img | 897 img | 832 img | 916 img |
| val | 471 (19.7 %) | 229 img | 251 img | 217 img | 250 img |
| test | 222 (9.3 %) | 110 img | 120 img | 102 img | 108 img |
| total | 2387 | 1188 img / 2369 boxes | 1268 / 2407 | 1151 / 1719 | 1274 / 2624 |
