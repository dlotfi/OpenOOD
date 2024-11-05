# MedOOD Dataset
MedOOD is a comprehensive collection of medical datasets designed to benchmark and evaluate out-of-distribution (OOD)
detection in the medical imaging domain. Building base on the [MedicOOD](https://github.com/benolmbrt/MedicOOD), MedOOD
includes curated datasets from diverse medical imaging modalities, providing a robust foundation for developing and
testing OOD detection models in real-world healthcare scenarios.


## Installation

#### Install requirements
```sh
pip install -r requirements.txt
```


## Usage
For preprocessing each dataset, you can run the following command:
```sh
python preprocess_NAME_OF_DATASET.py --base_dir BASE_DIR --output_dir OUTPUT_DIR --num_samples NUM_SAMPLES --seed SEED [--skip_existing] [--use_gpu]
```

### Arguments:
- base_dir (str, required): Base directory of the dataset.
- output_dir (str, required): Output directory of the processed data.
- num_samples (int, optional): Number of samples to process. Default is None, which samples all available data.
- seed (int, optional): Random seed for reproducibility. Default is None.
- skip_existing (flag): Skip re-processing existing files.
- use_gpu (flag): Use GPU for brain extraction. Only available for those pre-processing having the brain extraction step.

For generating transformation-shifted BraTS 2020 dataset, you can run the following command:
```sh
python transform_brats20.py --base_dir BASE_DIR --output_dir OUTPUT_DIR --seed SEED
```

For generating image list files for each processed dataset, you an run the following command:
```sh
python generate_imglist.py --input_dir INPUT_DIR --base_dir BASE_DIR --output_dir OUTPUT_DIR [--labels LABEL1 LABEL2 ...]
```

### Arguments:
- input_dir (str, required): Input directory of the processed data.
- base_dir (str, required): Base directory of the dataset.
- output_dir (str, required): Output directory for the generated image list files.
- labels (str, optional): Labels to include in the image list. Multiple labels can be specified.



## Datasets
### BraTS 2020
The [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) dataset provides a comprehensive collection of
pre-operative multimodal MRI scans of glioblastomas (GBM/HGG) and lower grade gliomas (LGG), sourced from multiple
institutions and pathologically confirmed. It includes updated 3T MRI scans with expert-labeled ground truths by
neuroradiologists. The dataset, available in NIfTI format (.nii.gz files), features native T1, post-contrast T1-weighted,
T2-weighted, and T2-FLAIR volumes, pre-processed for uniformity.

Pre-processing steps included converting DICOM files to NIfTI format, co-registering to the ([SRI24](https://www.nitrc.org/projects/sri24/))
anatomical template, resampling to an isotropic resolution of 1mm³, and skull-stripping.

We utilize only the [*Training*](https://www.cbica.upenn.edu/MICCAI_BraTS2020_TrainingData) split of this dataset, as
only the training data include labels. This split contains 369 multimodal MRI scans: 293 HGG and 76 LGG.

The T1-weighted MRIs serve as our in-distribution data, while the post-contrast T1-weighted and T2-FLAIR MRIs are used
as modality-shift out-of-distribution data. Additionally, we generate a transformation-shifted out-of-distribution
dataset by applying various transformations to the T1-weighted MRIs.


### LUMIERE
The [LUMIERE](https://springernature.figshare.com/collections/The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905)
dataset is a longitudinal collection of MRI data from 91 glioblastoma (GBM) patients, accompanied by expert
RANO (Response Assessment in Neuro-Oncology) evaluations. It includes a total of 638 study dates and 2487 images,
featuring T1-weighted pre- and post-contrast, T2-weighted, and fluid-attenuated inversion recovery (FLAIR) MRI sequences.
The dataset also provides pathology information, including MGMT methylation and IDH1 status, along with overall survival
times. Automated tumor segmentations from state-of-the-art tools (DeepBraTumIA and HD-GLIO-AUTO) and radiomic features
are also included.

We use the T1-weighted pre-operative MRIs in the LUMIERE dataset as a control dataset.

### BraTS 2023 - Pediatric Tumors
The [BraTS-PEDs 2023](https://www.synapse.org/Synapse:syn51514108) dataset is a multi-institutional, retrospective
cohort of multi-parametric MRI (mpMRI) scans, focused on pediatric high-grade gliomas, including diffuse midline
gliomas (DMGs) such as diffuse intrinsic pontine glioma (DIPG). The dataset comprises 228 pediatric cases collected
from institutions like the Children’s Brain Tumor Network (CBTN), Boston Children’s Hospital, and Yale University.
It includes conventional MRI sequences (T1, post-gadolinium T1-weighted, T2, and T2-FLAIR), pre-processed using the
same standardized pipeline as BraTS 2020 (NIfTI conversion, anatomical co-registration, and 1mm³ isotropic resampling).
The dataset is divided into training (99 patients), validation (45 patients), and testing cohorts, with ground truth
labels provided for the training cohort. Tumors were segmented into key subregions, including enhancing tumor (ET),
non-enhancing tumor (NET), cystic components (CC), and peritumoral edema (ED), supporting automated segmentation
algorithm development in pediatric neuro-oncology.

We use the T1-weighted MRIs as a population-shift out-of-distribution dataset.

### BraTS 2023 - Sub-Sahara-Africa
The BraTS 2023 Sub-Saharan Africa ([BraTS-SSA 2023](https://www.synapse.org/Synapse:syn51514109)) dataset is a publicly
available collection of multi-parametric MRI (mpMRI) scans from adult glioma patients across various imaging centers
in Sub-Saharan Africa, curated for the BraTS 2023 Challenge. This dataset uniquely represents pre-operative glioma
cases with T1-weighted, post-contrast T1-weighted, T2-weighted, and T2-FLAIR MRI volumes. Ground truth tumor
segmentations, including enhancing tumor, non-enhancing tumor core, and surrounding FLAIR hyperintensity, were refined
by radiology experts following automated pre-segmentation. It follows the BraTS 2020 pre-processing pipeline: NIfTI
conversion, common anatomical template co-registration, 1mm³ isotropic resampling, and skull-stripping.

We use the T1-weighted MRIs as a population-shift out-of-distribution dataset.

### WMH 2017
The [WMH 2017](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/AECRSD) dataset, used for the White Matter Hyperintensity (WMH) Segmentation Challenge, consists of brain MRI data
acquired from five different scanners across three hospitals in the Netherlands and Singapore. It includes 60 sets of
training images and 110 test images, each containing both 3D T1-weighted and 2D FLAIR images. The training data is
accompanied by manual expert annotations of WMH, provided as binary masks, and includes transformation parameters to
align the T1 images to the FLAIR images.

We use 3DT1 images, pre-processed with SPM12 r6685 to correct bias field inhomogeneities, as a diagnostic-shift
out-of-distribution dataset.

Since the official download source may be unreliable, you can use [this torrent](https://academictorrents.com/details/a6d90ae5a9ff4cc8184f122048495fd6bd18d6ba)
as an alternative for downloading the dataset.

### ATLAS v2.0
The [ATLAS v2.0](https://atlas.grand-challenge.org/Data/) dataset is a large, curated, open-source collection of
T1-weighted MRI scans and manually segmented stroke lesion masks, designed to improve automated lesion segmentation
algorithms. It builds on the previous ATLAS v1.2 release by expanding the number of datasets from 304 to 955 cases,
sourced from 33 research cohorts across 20 institutions worldwide, including a hidden test dataset for unbiased
algorithm evaluation.

We use T1-weighted MRIs from the ATLAS v2.0 dataset as a diagnostic-shift out-of-distribution dataset.

### EPISURG
The [EPISURG](https://rdr.ucl.ac.uk/articles/dataset/9996158?file=26153588) dataset is a curated collection of brain MRI
scans from 431 patients with drug-resistant focal epilepsy who underwent resective surgery at the National Hospital for
Neurology and Neurosurgery (NHNN) in London, UK. It includes 269 preoperative and 431 postoperative T1-weighted MR images,
with 133 of the postoperative scans manually annotated by expert raters to segment the resection cavities.

We use postoperative T1-weighted images where patients have a cavity in place of the epilepsy area as a diagnostic-shift
out-of-distribution dataset.

### IXI
The [IXI](https://brain-development.org/ixi-dataset/) dataset comprises approximately 600 MRI scans from healthy
subjects, collected as part of the Information eXtraction from Images (EPSRC GR/S21533/02) project. The dataset includes
various image types: T1, T2, PD-weighted, MRA, and diffusion-weighted images with 15 directions. Data was gathered from
three London hospitals using different MRI systems: a Philips 3T at Hammersmith Hospital, a Philips 1.5T at Guy’s
Hospital, and a GE 1.5T at the Institute of Psychiatry. The images are available in NIfTI format, along with demographic
information.

We use T1-weighted MRIs from the IXI dataset as a diagnostic-shift out-of-distribution dataset.

### CQ500
The [CQ500](http://headctstudy.qure.ai/dataset) dataset consists of 491 non-contrast head CT scans collected from
various radiology centers in New Delhi, India, and is divided into two batches: B1 (214 scans) and B2 (277 scans).
The dataset was curated to clinically validate deep learning algorithms for detecting critical findings such as
intracranial hemorrhage (ICH) and its subtypes (IPH, IVH, SDH, EDH, SAH), calvarial fractures, midline shift, and mass
effect. It was enriched for positive cases using a natural language processing (NLP) algorithm applied to clinical
radiology reports, ensuring a significant number of scans with these abnormalities. The dataset was reviewed by three
senior radiologists, and their majority consensus served as the gold standard for validation.

We use the CT scans from the CQ500 dataset as a modality-shift out-of-distribution dataset.

### CHAOS
The [CHAOS](https://chaos.grand-challenge.org/Data/) (Combined Healthy Abdominal Organ Segmentation) dataset is composed
of two databases: abdominal CT and MRI images, collected from DEU Hospital's PACS for the purpose of segmenting abdominal
organs. The CT database consists of images from 40 healthy liver donor patients, captured during the portal venous phase
using three different CT modalities. These CT scans have a resolution of 512x512 with an average of 90 slices per patient,
and the dataset presents challenges such as similar Hounsfield values between organs and anatomical variations. The MRI
database contains 120 DICOM datasets from T1-DUAL (in-phase and out-phase) and T2-SPIR sequences, with a lower resolution
of 256x256 and an average of 36 slices per patient. The MRI data, acquired using a 1.5T Philips scanner, is designed to
suppress fat and enhance the visibility of abdominal organs and their boundaries.

We use the MRI T1-DUAL in-phase and out-phase images as a far organ-shift out-of-distribution dataset.

### Lumbar Spine
The [Lumbar Spine](https://data.mendeley.com/datasets/k57fr854j2/2) dataset consists of clinical lumbar MRI scans from
568 symptomatic back pain patients, each accompanied by a diagnosis report from an expert radiologist. The dataset
includes MRI studies containing slices taken from sagittal and axial views, primarily focusing on the lowest three
vertebrae and intervertebral discs (IVDs). The number of slices per study ranges from 12 to 20 with resolutions typically
at 320x320 pixels, and pixel precision of 12-bit per pixel. Both T1-weighted and T2-weighted MRI scans are included,
with the dataset comprising a total of 1704 axial slices focusing on the last three IVDs. These slices are manually
labeled into four regions of interest (RoIs): the Intervertebral Disc (IVD), Posterior Element (PE), Thecal Sac (TS),
and the Area between Anterior and Posterior elements (AAP), which are crucial for detecting lumbar spinal stenosis.

We use T1-weighted MRIs from the Lumbar Spine dataset as a far organ-shift out-of-distribution dataset.
