阶段 1：Image Perception
(new-perception-note(2).ipynb)

原图 → 用 InternVL2.5 生成caption + object words → 对 object 做清洗和去重

阶段2：Image Segmentation
(segmentation(5).ipynb)
全部候选 label 做 segmentation：先把每个 object word 送进 Grounding DINO 出框，再把框给 SAM 出 mask (GroundedSAM)


阶段3：Inpaint
(inpaint-notebook.ipynb)