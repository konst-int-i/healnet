## Loading data

### Molecular data

### WSI

## Dataset descriptions

### TCGA-BLCA

The TCGA-BLCA dataset is a comprehensive characterisation of 409 bladder cancer samples. Available at 
https://portal.gdc.cancer.gov/projects/TCGA-BLCA.


Data categories: 

* Sequencing Reads: raw sequencing data generated from DNA or RNA sequencing experiments. It contains information about the nucleotide sequence of the DNA or RNA molecules.
* Transcriptome Profiling: information about the expression levels of genes in the bladder cancer samples. It is generated using RNA sequencing (RNA-seq) experiments that capture the transcriptome of the cancer cells.
* Simple Nucleotide Variation: single nucleotide variations (SNVs) or point mutations in the DNA sequence of the bladder cancer samples. It is generated using DNA sequencing experiments.
* Copy Number Variation: amplification or deletion of DNA segments in the bladder cancer samples. It is generated using DNA sequencing or microarray-based techniques.
* DNA Methylation: methylation status of DNA (i.e., epigenetic modification that plays a critical role in gene regulation and cellular function) in the bladder cancer samples. It is generated using methylation-specific DNA sequencing or microarray-based techniques.
* Clinical: clinical characteristics of the bladder cancer patients, such as age, sex, tumor stage, and survival outcomes.
* Biospecimen: biological samples used for the molecular analyses, such as the type of tissue or cell line, sample collection date, and quality control metrics.

Slide data: 
We download the slide data by filtering for the .svs file format via manifest. The manifest files we used are available in the data/manifests folder.

Note that for every case ID, there are multiple slides, which depend on the experimental strategy (tissue slide, diagnostic slide, etc.). We have been using those in accordance with 
the slide IDs from the molecular data files, which correspond to the diagnostic slides on the NIH portal. 

Note that the folder `data/tcga/gdc_manifests/full` contains both the tissue and diagnostic slides, while the folder `data/tcga/gdc_manifests/filtered` only contains the diagnostic slides.


Please see the data volume for each WSI dataset in the table below.

| Dataset | Data volume | N Samples | Slides per sample |
| --- |-------------|-----------|-------------------|
| TCGA-BLCA | 833 GB      | 412       | 2                 |
| TCGA-BRCA | 1.64 TB     | 1,098     | 3                 |

