# Label-Efficient Object Detection and Tracking in Military Drone Imagery: A Comparative Experimental Study of Supervised Detectors and Self-Supervised Backbones 

CSE 445 (Computer Vision), Group 02 Supervisor: Dr. Mohammad Rifat Ahmmad Rashid 

Md. Ariful Islam Opi – 2023-1-60-141 MD Shafin Ahamed Siam – 2023-1-60-007 Albin Israt Siddique – 2023-1-60-212 Nasrullah Kaisher Sijan – 2023-1-60-204 

Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh 

**_Abstract_ —Object detection in military drone imagery is constrained by the high cost of expert annotation, motivating two complementary questions addressed in this report. Part A asks which supervised detector architecture is best suited to the KIIT-MiTA dataset (seven classes of top-down aerial military hardware): we train and evaluate YOLOv10-s, YOLOv12-s, YOLOv26-s, and RF-DETR-nano, finding RF-DETR-nano the most accurate overall (test mAP50-95 = 0.526) but selecting YOLOv26-s (mAP50-95 = 0.460, 82 FPS) as the backbone carried into Part B, since Part B’s self-supervised weight-surgery pipeline requires a YOLO-family convolutional backbone that RF-DETR’s transformer architecture cannot provide. Part B asks whether self-supervised learning (SSL) pretraining on an 80% unlabelled image pool can reduce the labelled-data budget needed to reach competitive detection performance: we pretrain SimCLR, BYOL, I-JEPA, and a DINOv3-guided distillation on the unlabelled pool, transfer each backbone into YOLOv26s via weight surgery, and fine-tune at a 20% labelled budget. SimCLR wins this comparison (mAP50-95 = 0.0642), narrowly ahead of BYOL (0.0631) and I-JEPA (0.0601), with the DINOv3-distilled backbone trailing substantially (0.0396) due to an unresolved domain gap between its web-scale teacher and our narrow aerial domain. Critically, no SSL method approaches the COCOpretrained supervised baseline (0.2767) at the same label budget, and a five-point nested label-efficiency ablation (10–50% labels) shows this gap persisting rather than narrowing across the entire studied range, with break-even never reached. We deploy the winning SimCLR-initialised detector in a ByteTrack/BoT-SORT tracking pipeline over two video clips, evaluated with proxy metrics. We report these results, including the negative SSL finding, as evidence-based conclusions with a concrete practical recommendation: for this problem domain, additional annotation effort produces more downstream benefit than additional SSL pretraining engineering.** 

**_Index Terms_ —object detection, self-supervised learning, label efficiency, multi-object tracking, YOLO, RF-DETR, SimCLR, BYOL, I-JEPA, DINOv3, military imagery** 

## I. INTRODUCTION 

Detecting military hardware (tanks, artillery, radar installations, missile launchers, soldiers, and support vehicles) in top-down drone imagery is a problem for which annotated 

data is scarce by construction: the imagery is often restricted, the objects are rare, and correct labelling demands familiarity with military hardware silhouettes seen from an aerial vantage point that ordinary crowd-annotation pipelines cannot supply. A detector that performs well with a small labelled budget is therefore of direct practical value in this domain, and this report is organised around two experimental parts that jointly investigate how to obtain one. 

**Part A** asks a model-selection question: among modern supervised detector architectures, which is best suited to this dataset, under a fixed compute budget (Kaggle, dual T4 GPUs)? We train and evaluate four architectures, namely YOLOv10-s, YOLOv12-s, YOLOv26-s, and RF-DETR-nano, to establish both an accuracy ceiling and the specific backbone architecture that Part B builds on. 

**Part B** asks a label-efficiency question: if a detector’s backbone is pretrained on unlabelled imagery via self-supervised learning (SSL) before any labels are shown, how much labelled data can be saved relative to training from scratch or from a generic supervised (COCO) initialisation? We pretrain four SSL methods spanning three distinct learning paradigms, namely SimCLR (contrastive), BYOL (self-distillation without negative pairs), I-JEPA (masked latent prediction), and a DINOv3-guided teacher-student distillation, on an 80% unlabelled pool, transfer each into the Part A-selected YOLO backbone, and fine-tune at progressively smaller labelled budgets. 

Our contributions, stated as claims substantiated in Sections IV–VI, are: 

- RF-DETR-nano achieves the highest overall accuracy among four supervised detector architectures (mAP5095 = 0.526) but is architecturally incompatible with the SSL weight-surgery pipeline used in Part B, making YOLOv26-s (mAP50-95 = 0.460) the correct backbone choice for the label-efficiency study despite not being the single most accurate Part A model. 



<!-- Start of picture text -->
2200 Class Distribution (Instance1151 Count) Class ProportionVehicle<br>w 1000<br>=3 a0 5 tlaryia a> vn<br>5 600 Missile ZF<br>2255 “0200 n 487 386 M. RocketradarLauncher63Vag | Soldier<br>°<br>&sfSs #SNS &s ses&sé ¢ & “s= &oxe<br>’oa<br>Class.<br><!-- End of picture text -->



<!-- Start of picture text -->
a6 Part A: supervised detector architectures Part B: SSL backbones @ 20% labels<br>selected for 0526 0.30 0.2767<br>05 Part 8 backbone<br>0.460 025<br>0.421 0.432 "<br>wy8g 04 Pa8Fd 0.20<br>2° E015<br>02 0.10<br>oO 0.05 0.0682 0.0631 9.0601 0.0396 poe5<br>0.0 YOLOV10-s  YOLOV12-s — YOLOv26-s_—RF-DETR-n 0.00 SimCLR  BYOL —LJEPA_ DINOv3. Random-init pretrained_COCO-
<!-- End of picture text -->

training-side subsets only (never validation/test), targeting 60% of the majority class’s instance count. 

TABLE II: Effect of copy-paste augmentation on the SSL pool’s class balance (train-split instance counts, before vs. after; source: dataset-eda notebook). 

|**Class**|**Before**|**After**|
|---|---|---|
|Artillery (minority)|240|610|
|Radar (minority)|225|772|
|M. Rocket Launcher (minority)|311|705|
|Missile|362|764|
|Tank<br>|643|824|
|Soldier|907|937|
|Vehicle (majority)|881|1,033|
|Imbalance ratio (max:min)|4.03_×_|1.69_×_|
|Train images|1,339|2,377|


Table II quantifies this effect directly: targeted synthetic instances were added for the three minority classes (Artillery, Radar, M. Rocket Launcher) toward 60% of the pre-augmentation majority-class count (target = 544 instances/class), while every other class also gained some collateral instances as a side effect of pasting synthetic objects onto donor images that already contained other classes’ boxes. The net effect is that the majority-to-minority instance-count ratio falls from 4.03 _×_ before augmentation to 1.69 _×_ after, at the cost of growing the on-disk training image count from 1,339 to 2,377 (validation and test splits are left untouched, so this asymmetry never affects evaluation). 

An empirically-derived, dataset-wide decision, disabling rotation and vertical-flip augmentation (degrees=0.0, flipud=0.0), was applied uniformly across every training run in both Part A and Part B, because KIIT-MiTA’s consistent top-down aerial vantage point gives military hardware a canonical orientation that rotation destroys rather than usefully generalises over; we verified this empirically in Part A, where enabling rotation measurably reduced mAP. 

## _D. Hardware and fair-comparison protocol_ 

All experiments ran on Kaggle notebooks with two NVIDIA T4 GPUs. Part A’s four detectors were each trained under architecture-appropriate but otherwise consistent budgets (documented per-model in Section IV). Part B fixed every hyperparameter (epochs, batch size, image size, optimiser, augmentation policy) identically across all four SSL pretraining runs and all downstream fine-tuning runs (SSL-initialised and baseline alike), so that observed performance differences reflect the pretraining method rather than a confound in training budget. 

## _E. Evaluation metrics_ 

Detection is evaluated on the untouched test split using precision, recall, mAP50, and mAP50-95 (COCO-style, averaged over IoU thresholds 0.50–0.95), plus inference throughput (FPS) and parameter count for Part A’s architecture comparison. Tracking is evaluated with proxy metrics (Section V) in the absence of manually annotated ground truth: unique trackID count versus a manually-established expected object count, 

mean track length, fragmentation-event count, confidence distribution, and FPS. 

## III. METHODS 

## _A. Part A: supervised detector pipeline_ 

Four detector architectures were trained independently on the identical training/validation/test partition (Section II-B), each following a standard train–validate–test loop with a 1- epoch dry run first to verify the pipeline and estimate wallclock time before committing to a full run: 

- **YOLOv10-s** [4]: a single-stage, anchor-free CNN detector with consistent dual-label assignment for NMS-free inference. 

- **YOLOv12-s** [5]: an attention-centric YOLO variant incorporating area-attention and residual efficient-layeraggregation blocks. 

- **YOLOv26-s** [3]: the most recent Ultralytics YOLO release at the time of this study, used both as a Part A candidate and, once selected, as the backbone architecture for all of Part B. 

- **RF-DETR-nano** [6]: a DETR-family, transformer-based detector with a DINOv2-initialised backbone, trained with a considerably longer schedule (see Section IV) reflecting the slower convergence typical of transformer detectors relative to convolutional YOLO variants. 

The best-performing architecture by test mAP50-95 _among YOLO-family candidates_ was selected as the backbone for Part B (Section IV explains why RF-DETR, despite its higher raw accuracy, was not eligible for this role). 

## _B. Part B: self-supervised pretraining methods_ 

All four SSL methods pretrained a YOLOv26n backbone (yolo26n.yaml), or, for DINOv3, distilled into one, on the 1,339-image SSL pool, at 224 _×_ 224 resolution, batch size 32/GPU across two T4s, for 200 epochs, with mixed precision and a 10-epoch warmup wherever the method supports one, so that any downstream performance difference reflects the SSL objective itself rather than a training-budget confound. Pretraining, weight surgery, and evaluation used the ssl-detection-lab (ssldet) library [2] provided by the course instructor. 

TABLE III: SSL pretraining configurations. 

|**Parameter**|**SimCLR**|**BYOL**<br>**I-JEPA**|**DINOv3**|
|---|---|---|---|
|Teacher|None|EMA<br>EMA|ViT-B/16|
|Objective|NT-Xent|Bootstrap<br>Masked prediction|Distillation|
|Input size||224 _×_ 224||
|Batch size||32/GPU, 64 global||
|Hardware||2 _×_ NVIDIA T4||
|Epochs||200||
|Optimizer|AdamW|AdamW<br>AdamW|AdamW|
|Learning rate|3_×_10<sup>_−_4</sup>|1_×_10<sup>_−_4</sup><br>3_×_10<sup>_−_4</sup><br>|3_×_10<sup>_−_4</sup>|
|Weight decay||1_×_10<sup>_−_4</sup>||
|Warm-up|10|10<br>10|Native|
|Precision|FP16|FP16<br>FP16|16-mixed|
|Workers||2||
|Seed||42||



_1) SimCLR:_ SimCLR [7] contrasts two augmented views of the same image against all other images in the batch via the NT-Xent loss, 



with temperature _τ_ = 0 _._ 20. Negative pairs are essential: without them, mapping every image to a constant embedding trivially minimises the positive-pair term, causing collapse. The projection head is discarded after pretraining; only the backbone encoder is transferred downstream (Section III-C), since the projection head specialises for instance discrimination and discards detail useful for detection. _2) BYOL:_ BYOL [8] removes the need for negative pairs via an asymmetric online/target network pair, the target updated only by an EMA of the online weights ( _τ_ ema = 0 _._ 996 _→_ 1 _._ 0). The online predictor learns to predict the EMA target’s projection of a second view, with stop-gradient on the target branch. Collapse is avoided through three mechanisms acting jointly: the stop-gradient prevents trivial copying; the predictor’s one-sided asymmetry destabilises the constant-output solution; and batch normalisation induces cross-example interaction within a batch. Our pretraining run reflects a stabilised configuration reached after early attempts at alternative momentum schedules exhibited collapse. 

_3) I-JEPA:_ I-JEPA [9] predicts masked target-block representations directly in embedding space rather than reconstructing pixels, avoiding the capacity waste of pixel-level reconstruction on high-frequency, semantically-irrelevant detail. Target blocks (4 blocks, scale [0 _._ 10 _,_ 0 _._ 25], aspect ratio [0 _._ 75 _,_ 1 _._ 50]) are masked out of the context block so the prediction task cannot be solved trivially by copying. Because the masking-and-prediction task itself supplies an invariance signal, I-JEPA requires no hand-crafted augmentation policy, a property that sidesteps the dataset-specific augmentation failures we observed elsewhere (e.g., rotation). 

_4) DINOv3-guided distillation:_ Unlike the other three methods, our fourth branch distils knowledge from a _frozen_ , already-pretrained DINOv3 ViT-B/16 teacher [10] (pretrained at web scale on LVD-1689M) into a randomly-initialised YOLOv26n student, both observing our own SSL pool, under a feature-matching distillation loss (AdamW, lr 3 _×_ 10<sup>_−_4</sup> ) for the same 200-epoch budget. This is domain-adaptive distillation rather than pretraining from scratch: the student is guided at every step by a teacher that already encodes rich, generalpurpose visual features from a corpus far larger than our pool. This asymmetry matters for the fairness of our four-way comparison (Section V) and is revisited in the Discussion. 

## _C. Weight surgery and downstream fine-tuning_ 

Each SSL-pretrained encoder is transferred into a full YOLOv26n detector by loading its backbone weights into the corresponding detector backbone layers by name ( _weight surgery_ ), verifying _≥_ 95% of backbone parameter tensors match by shape; the neck and head, with no SSL counterpart, retain standard random initialisation. Every fine-tuning 

run (all four SSL-initialised detectors, plus random-init and COCO-pretrained baselines) then follows an identical twostage protocol at the 20% labelled subset: (1) a 10-epoch head warm-up with the backbone frozen (lr 1 _×_ 10<sup>_−_3</sup> ), and (2) 140 epochs of full fine-tuning (lr 3 _×_ 10<sup>_−_4</sup> , cosine decay, close_mosaic=30, patience 15), image size 640, batch 32, AdamW, rotation/flip disabled. 

## _D. Tracking pipeline_ 

The winning SSL-initialised detector (Section V) is deployed in a tracking-by-detection pipeline: the detector runs independently on every video frame, and ByteTrack [11] associates detections across frames into persistent identity tracks, using a Kalman filter to predict each track’s position before matching. ByteTrack’s key mechanism is two-stage association: high-confidence detections are matched first, and only then are remaining unmatched tracks matched against low-confidence detections that a naive tracker would discard, which is critical for surviving brief confidence drops under occlusion or motion blur without breaking the track. We additionally run BoT-SORT [12] (adding appearance reidentification and camera-motion compensation) on one clip for comparison. Because ground-truth track annotation was not performed, we evaluate with proxy metrics rather than MOTA/IDF1 (Section V). 

## IV. RESULTS: PART A 

Table IV reports the consolidated test-set comparison across all four detector architectures. 

TABLE IV: Part A: consolidated detector comparison, test split. 

|**Model**|**mAP50**|**mAP50-95**|**Prec.**|**Rec.**|**F1**|**FPS**|**Params(M)**|
|---|---|---|---|---|---|---|---|
|YOLOv10-s|0.646|0.421|0.673|0.600|0.635|84.7|7.22|
|YOLOv12-s|0.665|0.432|0.631|0.636|0.633|71.2|9.23|
|**YOLOv26-s**|0.648|**0.460**|0.722|0.624|0.669|82.1|9.47|
|RF-DETR-n|**0.754**|**0.526**|0.724|0.706|0.715|20.6|30.45|



Fig. 3: Test-set mAP50 and mAP50-95 across all four Part A detector architectures. 

## _A. Model-selection justification_ 

RF-DETR-nano is the most accurate model overall (mAP5095 = 0.526, +14% relative to the next-best YOLOv26-s), consistent with the general trend that DETR-family, transformerbased detectors with strong pretrained backbones (RF-DETR’s 



<!-- Start of picture text -->
FPS ve mar Latency vs Accuracy Parameters vs maP<br>ose ° os2 ° 032 °<br>50 050 030<br>Boas.5 HiBoss FiBows.<br>cas Ce ee a aus] rons<br>owe on ae on oe qo<br>oz e » @ %cf & % ® ep on] » omeS » S tater)» S&S & & on] oor® B ara)Fa ae)
<!-- End of picture text -->



<!-- Start of picture text -->
YoLov26-s — Robustness<br>07 Robustness——,— mAP@50 vs corruption severity 100 Accuracy retained (sev=2)<br>06 = |. 7<br>os i&<br>z os £0H<br>03 |pon*S- Sa Seecoteron gFx<br>oo bt 3 i severity 3 5 5 ose ooos re oss e<br>* * a<br><!-- End of picture text -->



<!-- Start of picture text -->
YOLOv26-s — Custom confusion matrix @ conf=0.25<br>Confusion matrix (counts) Confusion matrix (row-normalized)<br>wiay fe 6 8 8 ola ars. o<br>a6<br>Emrocketiancrer}2 © 4 1 wm 8 8 ee 7  m.rocketLauncher]2 co enn 00 oom aoa os04<br>i sir] see . -—& 7 3 setter] so) am =~ « & 03<br>a2<br>weigond]Vehicle} o© °8 18 ° mery a 20 tactgroundVehicle} | 200cos 000ome optoot cur om(MI ane ERI6 a6en oa<br>ase S$- a se ard ea#es- o & o&se #¢a ea weese #&¥ > ao<br>od &<br>mredicted Predicted
<!-- End of picture text -->



<!-- Start of picture text -->
YOLOv26-s — Qualitative results (GT vs prediction)<br>GT — image_s312_kit_1367 (1 obj) PRED— 1 det<br>GT — image_5312_kit_1408 (2 obj) PRED — 2 det<br>TSG? “eS TIT RI? Ge<br>ie f te ra mf | FI<br>Fa // CSAeee = le <1 /x eeea Re =<br>2 | ft’ | Pea "9 P . = d } Pag, wm<br>ie Loeq ij“f | -jia) —- CSe!ay CUE+ 7 iy eeay*-B<br>aay ‘
<!-- End of picture text -->



<!-- Start of picture text -->
SIMCLR training loss (NT-Xent)<br>4.0 i<br>3.5 i<br>382253.02.0 1®\%:%<br>®<br>0.5151.0 te“ce“Fc,ung,“ ret eceegteccacaee LCHEEELE LEEMEEEERO<br>0 2 50 75 100 125 150 175 200<br>Epoch<br><!-- End of picture text -->



<!-- Start of picture text -->
t-SNE of SIMCLR SIMCLR validation embeddings, coloured by dominant class<br>° ©@@ Attilary M. M. Rocket Launcher<br>ee . e@©© MissileRadarRadar<br>10 oor ee @ Soldier<br>° ° ° mal<br>e ° @ Vehicle<br>ee e<br>se °<br>- . ° e ° I.
<!-- End of picture text -->



<!-- Start of picture text -->
BYOL training loss Targetnetwork momentum momentum<br>175 + 1.0000 a<br>150 0.9995
<!-- End of picture text -->



<!-- Start of picture text -->
BYOL training loss Targetnetwork momentum momentum t-SNE of SIMCLR SIMCLR validation embeddings, coloured by dominant class<br>175 + 1.0000 a<br>150 0.9995 ° ©@@ Attilary M. M. Rocket Launcher
<!-- End of picture text -->



<!-- Start of picture text -->
WJEPA latent-prediction loss (Smooth L1) Target-network momentum<br>oS " 0.9995,1.00001.0000 -——
<!-- End of picture text -->



<!-- Start of picture text -->
DINOv3-guided distillation loss<br>6.5
<!-- End of picture text -->



<!-- Start of picture text -->
SIMCLR -- confusion matrix (test set)<br>Confusion Matrix Normalized<br>Artilary 018 007 «006-02 00709 06<br>Radar 009 021 © «006-«S 0.02005. 003006 4
<!-- End of picture text -->



<!-- Start of picture text -->
o2s Four SSL methods @ 20% labels Ranking -- winner: SIMCLR<br>fecal<br>0.20 = ee| simcur ogee<br>03s ByOL 0.0630<br>&0.10 EPA 0.0601<br>0.05 pinova 0.0396<br>0.00 SIMCLR, BYOL EPA DINov3 0.00 oor 0.02 0.03 0.08 0.05 0.08<br>Test mAP50.95<br><!-- End of picture text -->



<!-- Start of picture text -->
o Peuve to frconidenceSIMCLR ~ test set curves (macro-averageto  over clases)recison-contdence to Recal.conidence<br>i © i ;<br><!-- End of picture text -->



<!-- Start of picture text -->
youtube<br>S ons ~ Pee eee a eee ~<br>. Ses Po ad ae, - Ss<br>4 4 =. a,<br>7S 22S ;<br>ai_generated
<!-- End of picture text -->



<!-- Start of picture text -->
3023 {jm unique tacks Unique track IDs vs. expected count Confidence distribution over frames T= youtube<br>40<br>20<br>5 é€30<br>youtube ai_generated 03 04 05 06 07 08 09<br><!-- End of picture text -->



<!-- Start of picture text -->
rho=0.10 fiLy r,
<!-- End of picture text -->



<!-- Start of picture text -->
Label-efficiency curve<br>0.4<br>in 03<br>23}a2<<br>—<br>%<br>@ 0.2<br>O01<br>—®— SimCLR (best SSL backbone)<br>=—®— COCO-pretrained (supervised baseline)<br>--- Part A, 100% labels (mAP50-95=0.460)<br>10 15 20 25 30 35 40 45 50<br>Label fraction (%)<br><!-- End of picture text -->

(Figure 18) shows the opposite: the gap remains wide and nonnarrowing across the entire 10–50% range. This is the report’s central, honestly-reported negative finding: SSL pretraining confined to a small (1,339-image), narrow-domain, singledataset pool does not substitute for large-scale supervised (COCO) pretraining in this problem setting, at any label budget we tested. 

**Why did DINOv3 under-perform despite its more powerful teacher?** We attribute this to two gaps that had to be bridged simultaneously within the shared 200-epoch budget: a domain gap (DINOv3’s web-scale teacher distribution vs. our narrow aerial-military domain) and an architecture gap (ViT teacher, CNN student). The other three methods face no analogous gap, since they train a YOLOv26n encoder endtoend on our own pool without an external teacher’s domain mismatch to overcome. We regard this as evidence that distillation budget must scale with domain gap, not merely with epoch count matched to competing methods. This is directly supported by DINOv3’s loss curve (Figure 11), which is still trending downward at epoch 200, suggesting the distillation had not yet converged within budget. 

**How did detector quality propagate into tracking?** The tracking pipeline (Section V) inherits the SimCLRinitialised detector’s modest absolute mAP50-95 (0.0642). Because tracking-by-detection has no independent objectrecognition capability of its own, every missed or spurious per-frame detection directly threatens track continuity; the proxy metrics (Figure 17) are consistent with this propagation, reinforcing that Part B’s central finding (weak SSL backbones on this domain) is not confined to the detection metric table but visibly affects every downstream consumer of the detector, including tracking. 

**Practical annotation-budget recommendation.** For a team facing a similarly narrow-domain, small-scale detection problem and choosing between investing effort in an SSL pretraining pipeline versus simply annotating more images with a COCO-transfer baseline, our results favour the latter without qualification within the range we tested: every additional labelled percentage point produced a larger downstream mAP5095 gain under COCO initialisation than switching from COCO to SimCLR initialisation ever could. We report this as an actionable, domain-specific recommendation rather than a general claim about SSL’s value. 

## VII. LIMITATIONS AND FUTURE WORK 

**Compute-constrained pretraining.** All four SSL methods were pretrained for a fixed 200 epochs on two T4 GPUs; a substantially longer budget, particularly for DINOv3’s distillation (whose loss curve had not yet plateaued, Figure 11), might change its relative standing. 

**Single dataset, single domain.** All findings are specific to KIIT-MiTA’s narrow, top-down military aerial domain and its particular scale (1,681 images); we make no claim that SSL pretraining is generally weak, only that it under-performed supervised pretraining _in this setting_ . 

**Synthetic-video caveat.** One of our two tracking clips is AI-generated; synthetic footage can differ systematically from real footage in texture, motion, and lighting, so tracking results on that clip are reported alongside, not in place of, the realfootage clip’s results. 

**What a larger study would add.** A 1%-label extension of the label-efficiency ablation would test whether SSL’s theoretical advantage emerges at the most extreme low-label regime even though it did not emerge across 10–50%; a larger unlabelled pool (beyond our 1,339-image SSL pool) would test whether our central negative finding is a function of pool size specifically, or of domain narrowness more fundamentally. These remain open questions this report’s compute budget could not resolve. 

## VIII. CONCLUSION 

Among four SSL pretraining methods SimCLR is the strongest (mAP50-95 = 0.0642 at 20% labels), but no SSL method, including SimCLR at any tested label fraction up to 50%, approaches the COCO-pretrained supervised baseline (0.2767 at 20%; 0.3761 at 50%), and this gap does not narrow as theory predicts. We conclude that for narrowdomain, small-scale detection problems structurally similar to military drone imagery, annotation investment is a more reliable path to downstream performance than small-scale SSL pretraining engineering, and we deploy and report our best available SSL-initialised detector’s tracking performance honestly, including its detector-quality-limited proxy metrics, rather than substituting a stronger but non-SSL checkpoint. 

## REFERENCES 

- [1] KIIT-MiTA Dataset, Mendeley Data. [Online]. Available: https://data. mendeley.com/datasets/drjmrf5kk5/1 

- [2] M. R. A. Rashid, “ssl-detection-lab,” 2024. [Online]. Available: https: //github.com/rifat963/ssl-detection-lab 

- [3] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLO,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics 

- [4] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, “YOLOv10: Real-time end-to-end object detection,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , 2024. 

- [5] Y. Tian, Q. Ye, and D. Doermann, “YOLOv12: Attention-centric realtime object detectors,” _arXiv preprint arXiv:2502.12524_ , 2025. 

- [6] Roboflow, “RF-DETR: A real-time, DETR-based object detection model,” 2025. [Online]. Available: https://github.com/roboflow/rf-detr 

- [7] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in _Proc. ICML_ , 2020, pp. 1597–1607. 

- [8] J.-B. Grill, F. Strub, F. Altch´e, C. Tallec, P. H. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar, B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko, “Bootstrap your own latent,” in _Advances in Neural Information Processing Systems (NeurIPS)_ , vol. 33, 2020, pp. 21271–21284. 

- [9] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in _Proc. CVPR_ , 2023, pp. 15619–15629. 

- [10] O. Sim´eoni _et al._ , “DINOv3,” _Meta AI Research_ , 2024. 

- [11] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, “ByteTrack: Multi-object tracking by associating every detection box,” in _Proc. ECCV_ , 2022, pp. 1–21. 

- [12] N. Aharon, R. Orfaig, and B.-Z. Bobrovsky, “BoT-SORT: Robust associations multi-pedestrian tracking,” _arXiv preprint arXiv:2206.14651_ , 2022. 

## IX. CONTRIBUTION 

This statement lists which group member led which notebook(s) and report section(s). 

|**Member**|**Notebooks / Report Sections Led**|
|---|---|
|Md. Ariful Islam Opi<br>(2023-1-60-141)|_EDA Notebook; Notebook 1a/1b – SimCLR and Notebook 5 – Tracking; Part A Notebook –_<br>_RF-DETR; Report Sections III-B1, V-C_|
|MD Shafn Ahamed Siam<br>(2023-1-60-007)|_Notebook 2a/2b – BYOL; Part A Notebook – YOLOv10; Report Section III-B2_|
|Albin Israt Siddique<br>(2023-1-60-212)|_Notebook 3a/3b – I-JEPA; Part A Notebook – YOLOv12; Report Section III-B3_|
|Nasrullah Kaisher Sijan<br>(2023-1-60-204)|_Notebook 4a/4b – DINOv3; Bonus Notebook; Part A Notebook – YOLOv26; Report Section_<br>_III-B4_



