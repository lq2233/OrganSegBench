# OrganSegBench
OrganSegBench: Benchmarking and Enhancing Segmentation Foundation Models for Clinical Reliability
- Organ segmentation via foundation models
- Model performance evaluation
- Segmentation Error Distribution Computation
- Multi-source Knowledge Distillation
## 1. Organ segmentation via foundation models - How to start

![SFM description](https://github.com/lq2233/OrganSegBench/blob/main/Model_description.png)

### Step 1: Deploy the foundation models in local devices

|Foundation models|Backbone|Accessing URL|
|---|---|---|
|SAM|Vision Transformer|https://github.com/facebookresearch/segment-anything|
|SAM2|Hierarchical Transformer|https://github.com/facebookresearch/sam2|
|MedSAM|Vision Transformer|https://github.com/bowang-lab/MedSAM|
|MedSAM2|Hierarchical Transformer|https://github.com/bowang-lab/MedSAM2|
|SAT|UNet|https://github.com/zhaoziheng/SAT|
|TotalSegmentator|nnUNet|https://github.com/wasserth/TotalSegmentator|
### Step 2: Download the checkpoint for each foundation models

|Foundation models|Checkpoint|Accessing URL|
|---|---|---|
|SAM|ViT-H SAM|https://github.com/facebookresearch/segment-anything|
|SAM2|SAM2.1_HIERA_LARGE|https://github.com/facebookresearch/sam2/tree/main?tab=readme-ov-file|
|MedSAM|MedSAM_VIT_B|https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN|
|MedSAM2|SAM2.1_HIERA_TINY|https://github.com/bowang-lab/MedSAM2?tab=readme-ov-file|
|SAT|SAT-PRO|https://huggingface.co/zzh99/SAT/tree/main/Pro|
|TotalSegmentator|TotalSegmentator V2|https://github.com/wasserth/TotalSegmentator|

### Step 3: Put them into the directory
According to the requirements, download the checkpoint of each foundation model and place it in the required locations.

### Step 4: Performing organ segmentation

Prepare data for the foundation models: use 3D data (NII) for SAT and TotalSegmentator, and use 2D data for SAM-like models. The 2D data (png) can be obtained by slicing the 3D data along the z-axis.
![SFM](https://github.com/lq2233/OrganSegBench/blob/main/Pipeline_SFMs.png)

If you only want to run a single experiment on a model, execute the following command.
For SAM2 or other other SAM-like models,install the needed packages as:
```
cd ./Organ_Segmentation_via_Foundation_Models/SAM2
pip install -r requirement.txt
```

```
cd ./Organ_Segmentation_via_Foundation_Models/SAM2
python SAM2.py
```
The segmenttaion results for instances are shown as follows, where **a** denotes the source image, **b** denotes the annotation, **c** denotes the bounding box for segmentation and **d** denotes the segmentation masks.
![Segmentation_via_SAM2](https://github.com/lq2233/OrganSegBench/blob/main/Segmentation_via_SAM2.png)

For SAT, install the needed packages as:
```
cd ./Organ_Segmentation_via_Foundation_Models/SAT
pip install -r requirement.txt
```
Generate the **json** file as ```chest_demo_CT.jsonl``` or ```abdomen_demo_MRI.jsonl``` via ```jsonl.ipynb```.

Run the segmentation process as follows.
```
cd ./Organ_Segmentation_via_Foundation_Models/SAT
torchrun --nproc_per_node=1 --master_port 1234 inference.py --rcd_dir './demo/inference_demo/results' --datasets_jsonl './abdomen_demo_MRI.jsonl' --vision_backbone 'UNET-L' --checkpoint './SAT-Pro-checkpoints/SAT_Pro.pth' --text_encoder 'ours' --text_encoder_checkpoint './SAT-Pro-checkpoints/text_encoder.pth' --max_queries 256 --batchsize_3d 1
```

Due to its large size, the SAT-Pro checkpoint file is not included in this repository. You can download it from the following sources:
* [Here](https://pan.baidu.com/s/1YAAebtMBHsBxxhMQl4NEvA?pwd=m663)
* [Huggingface](https://huggingface.co/zzh99/SAT/tree/main/Pro)

For TotalSegmentator, install the needed packages as:
```
cd ./Organ_Segmentation_via_Foundation_Models/TotalSegmentator
pip install -r requirement.txt
```
Run the segmentation process as follows.
```
cd ./Organ_Segmentation_via_Foundation_Models/TotalSegmentator
python TotalSegmentator.py
```

## 2. Model performance evaluation - How to start
We comprehensively evaluate the performance of SFMs in multiple dimensions, including accuracy, robustness, fairness, generalization and clinical utility. Our preliminary work (on the fairness of foundation models) has been published at **[MICCAI 2024](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_41)**.
### Step 1: Accuracy assessment
The DSC and 95%HD were used as the indexes for accuracy assessment. 
Given ```seg``` as the segmentation masks and '''ann''' as the annotation, the DSC and 95%HD can be quantified as follows.
```
from medpy.metric.binary import dc, hd95
def Accuracy_assessment()
  DSC = dc(seg, ann)
  HD95 = hd95(seg, ann)

  return DSC, HD95
```
### Step 2: Robustness assessment
We assessed model performance on the most challenging cases (cases with lower 5% performance).
Given DSC_group and HD95_group as the performance lists for the subjects on specific organ segmentation, to identify the cases, the follow code can be used.
```
import numpy as np
def Cases_identify(DSC_group, HD95_group, metric='DSC'):
  n = max(1, int(len(scores) * 0.05))
  Idx_DSC_lower5 = np.argsort(scores)[:n]
  Idx_HD95_lower5 = np.argsort(scores)[::-1][:n]

  return Idx_DSC_lower5, Idx_HD95_lower5
```
### Step 3: Fairness assessment
To investigate algorithmic bias, we stratified the cohort by gender (male, female), age (20-30, 30-40, 40-50, 50-60 years), and BMI (>21, 21-24, <24). 

For each sensitive attribute, a one-way analysis of variance (ANOVA) was applied to the DSC and HD95 distributions to test for statistically significant performance differences among subgroups.

Given the ```Male_organ```, ```Female_organ``` as the performance lists grouped by sensitive attribute **gender** on the specific organ segmentation, to assess fairness, the follow code can be used.
```
from scipy import stats
def fairness_assessment():
  result = []
  for organ in organs:
    f_statistic, p_value = stats.f_oneway(Male_organ, FeMale_organ)
    result.append(p_value)

  return result
```
To assess the overall fairness, the ratio of **N.S (Not Significant, higher fairness)*** among organs was quantified.
```
def Overall_fairness():
  result = fairness_assessment()
  count_NS = 0
  for i in result:
    if i>0.05:
      count_NS = count_NS+1

  return count_NS/len(result)
```
### Step 4: Generalization assessment
As for generalization, we quantified the variance of accuracy among organs via standard deviation.
Given the ```organs``` as the organ list, ```seg_organ``` as the segmentation masks for the specific organ, '''ann_organ''' as the annotation for the specific organ, the key code is as follows for generalization assessment.
```
from medpy.metric.binary import dc, hd95
import numpy as np
def Generalization_assessment(organs, seg_organ, ann_organ):
  DSC_list = []
  HD_list = []
  for i, organ in enumerate(organs):
    DSC_list.append(dc(seg_organ[i], ann_organ[i]))
    HD_list.append(hd95(seg_organ[i], ann_organ[i]))
  Gen_DSC = np.std(DSC_list)
  Gen_HD = np.std(HD_list)

  return Gen_DSC, Gen_HD
```
### Step 5: Clinial utility assessment
For cardiac analysis, we computed nine phenotypes, including LVEDV (Left Ventricle End-Diastolic Volume), RVEDV (Right Ventricle EndDiastolic Volume), LVESV (Left Ventricle End-Systolic Volume), RVESV (Right Ventricle End-Systolic Volume), LVSV (Left Ventricular Stroke Volume), RVSV (Right Ventricular Stroke Volume), LVEF (Left Ventricular Ejection Fractions), RVEF (Right Ventricular Ejection Fractions) and LVCO (Left Ventricular Cardiac Output).

Denote $$V_{EDLV}$$, $$V_{EDRV}$$, $$V_{ESLV}$$, $$V_{ESRV}$$ as the volume (mL) of the region EDLV, EDRV, ESLV and ESRV of cardiac respectively, Then:
|Phenotypes|
|---|
|$$LVEDV = V_{EDLV}$$|
|$$LVESV = V_{ESLV}$$|
|$$RVEDV = V_{EDRV}$$|
|$$RVESV = V_{ESRV}$$|
|$$LVSV = LVEDV - LVESV$$|
|$$RVSV = RVEDV - RVESV$$|
|$$LVEF =\frac{LVSV}{LVEDV}×100$$|
|$$RVEF =\frac{RVSV}{RVEDV}×100$$|
|$$LVCO = LVSV×alpha×10^-3$$|
|$$alpha = 60.0×\frac{cycle}{duration}$$|

The $$\frac{cycle}{duration}$$ represents the number for complete heartbeat round in a second and can be read from the header file of the ‘NIFTI’ data.

For other organs, we calculated the volume from the groundtruth mask ($$VGT$$) and the model’s segmentation ($$\overline{V}$$) to derive
the Volume Error Rate, $$VER = \frac{|VGT - \overline{V}|}{VGT}$$.

## 3. Segmentation Error Distribution Computation - How to start
In contrast to the traditional reliance on metrics such as DSC and HD, we analyzed model performance by examining the overall spatial distribution of segmentation errors, which is crucial for downstream tasks and the clinical applicability of SFMs.
### Step 1: Install the needed packages
Install the needed packages.
```
cd ./Segmentation_Error_Distribution_Computation
pip install -r requirement.txt
```
### Step 2: Data processing
Transform the Nifit format data into the STL format data, further into the point cloud.
```
cd ./Segmentation_Error_Distribution_Computation
python NII2STL.py
python STL2PointCloud.py
```
### Step 3: Computing segmentation error distribution per subject
```
cd ./Segmentation_Error_Distribution_Computation
python Distance.py
```
### Step 4: Computing the overall segmentation error distribution among subjects
All above 4 steps are integrated in to the main code, as shown in follows:
```
cd ./Segmentation_Error_Distribution_Computation
python Main.py
```
![Error_distribution](https://github.com/lq2233/OrganSegBench/blob/main/4_1.png)

## 4. Multi-source Knowledge Distillation - How to start
To improve the performance and cost-effectiveness of SFMs for clinical deployment, we introduce two ensemble strategies: training-free fusion and multi-source knowledge distillation. This work extends our **[MICCAI 2025](https://link.springer.com/chapter/10.1007/978-3-032-04937-7_19)** publication, where we choose **UNet** (CNN-base) and **HSNet** (ViT-base) as the student models for **Deterministic Distillation** and **Probabilistic UNet** for **Probabilistic Distillation**.


![Error_distribution](https://github.com/lq2233/OrganSegBench/blob/main/Distillation.png)


### Step 1: Deterministic Distillation

#### 1. install needed packages for the model deployment.
For UNet,
```
cd ./Multi-source_Knowledge_Distillation/Deterministic_Distillation/UNet/
pip -r requirement.txt
```

For HSNet,

```
cd ./Multi-source_Knowledge_Distillation/Deterministic_Distillation/HSNet/
pip -r requirement.txt
```

#### 2. Prepare the training/test set.

Using pancreas as the instance, For training set, collect the AVG fusion of masks generated by 6 SFMs as the training label. 

The training set should be arranged as follows.
```bash
./Multi-source_Knowledge_Distillation/Deterministic_Distillation/HSNet/DataSet_Segmentation/TrainDataset-pancreas/
├── images/
│   ├── 0.png
│   └── 1.png
└── masks/
    ├── 0.png
    └── 1.png
```
For test set, the manual annotations are collected for performance evaluation.

The test set should be arranged as follows, where **ID** is used to identify the subject for further performance assessment.

```bash
./Multi-source_Knowledge_Distillation/Deterministic_Distillation/HSNet/DataSet_Segmentation/TestDataset-pancreas/
├── images/
│   ├── ID_0.png
│   └── ID_1.png
└── masks/
    ├── ID_0.png
    └── ID_1.png
```
#### 3. Training the student model
If you finished to prepare the dataset, you can execute the following command for training.
For UNet,
```
cd ./Multi-source_Knowledge_Distillation/Deterministic_Distillation/UNet/
python training_model.py
```
For HSNet,
```
cd ./Multi-source_Knowledge_Distillation/Deterministic_Distillation/HSNet/
python training_model.py
```
#### 4. Valid the trained student model

For UNet，using the checkpoint provided in ```./Multi-source_Knowledge_Distillation/Deterministic_Distillation/UNet/model_path_EDRV_for_test/EDRV/``` as an example，you can execute the folloewing command for validation.
```
cd ./Multi-source_Knowledge_Distillation/Deterministic_Distillation/UNet/
python test.py
python DSC_Calculation.py
python HD_Calculation.py
```
For HSNet，using the checkpoint provided in ```./Multi-source_Knowledge_Distillation/Deterministic_Distillation/HSNet/model_pth_pancreas_for_test/PolypPVT_pancreas/``` as an example，you can execute the folloewing command for validation.
```
cd ./Multi-source_Knowledge_Distillation/Deterministic_Distillation/HSNet/
python test.py
python DSC_Calculation.py
python HD_Calculation.py
```

### Step 2: Probabilistic Distillation

#### 1. install needed packages for the model deployment.
```
cd ./Multi-source_Knowledge_Distillation/Probabilistic_Distillation/Probabilistic_UNet/
pip -r requirement.txt
```
#### 2. Prepare the training/test set.

Using Aorta(AO) as the instance, for training set, collect the masks generated by 6 SFMs as the training label. 

The training set should be arranged as follows.
```bash
./Multi-source_Knowledge_Distillation/Probabilistic_Distillation/Probabilistic_UNet/Data_Sample/TrainDataset/AO/
├── /ID_0
│   ├── images
│      └── image_ID_0.png
│   └── masks
│      ├── SAM_label_ID_0.png
│      ├── SAM2_label_ID_0.png
│      ├── MedSAM_label_ID_0.png
│      ├── MedSAM2_label_ID_0.png
│      ├── SAT_label_ID_0.png
│      └── TotalSeg_label_ID_0.png
└── /ID_1
│   ├── images
│      └── image_ID_1.png
│   └── masks
│      ├── SAM_label_ID_1.png
│      ├── SAM2_label_ID_1.png
│      ├── MedSAM_label_ID_1.png
│      ├── MedSAM2_label_ID_1.png
│      ├── SAT_label_ID_1.png
│      └── TotalSeg_label_ID_1.png
```
For test set, the manual annotations are collected for performance evaluation.

The test set should be arranged as follows.
```bash
./Multi-source_Knowledge_Distillation/Probabilistic_Distillation/Probabilistic_UNet/Data_Sample/TestDataset/AO/
├── /ID_0
│    ├── image_ID_0.png
│    └── label_ID_0.png
└── /ID_1
│   ├── image_ID_1.png
│   └── label_ID_1.png

```

#### 3. Training the student model
If you finished to prepare the dataset, you can execute the following command for training.
```
cd ./Multi-source_Knowledge_Distillation/Probabilistic_Distillation/Probabilistic_UNet/
python Training_Model.py
```
#### 4. Valid the trained student model
Using the checkpoint provided in ```./Multi-source_Knowledge_Distillation/Probabilistic_Distillation/Probabilistic_UNet/Model_path_test/AO/``` as an example，you can execute the folloewing command for validation.
```
cd ./Multi-source_Knowledge_Distillation/Probabilistic_Distillation/Probabilistic_UNet/
python Test.py
```

## License
The code is licensed under the MITlicense.

## Citation
If you wang to cite this study, please use the following BibTeX entry.
```
@article{wang2025organsegbench,
  title={OrganSegBench: A Comprehensive Multi-Organ Benchmark for Segmentation Foundation Models with a Practical Synergy Pathway to Clinical Application},
  author={Li, Qing and Zhang, Yizhe and Guo, Xin and Zhang, Haosen and Li, Yan and Yang, Mo and Zhang, Yajing and Sun, Mengting and Sun, Longyu and others},
  year={2025}
}
```
