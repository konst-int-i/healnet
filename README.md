# x-perceiver

Explainable perceiver for multimodal tasks

## Setup 

Install or update the conda environment using and then activate

### Conda
```
conda env update -f environment.yml
conda activate cognition
```

### Command line installation

On Mac or Linux, you can install the below dependencies using the command line

```bash
invoke install --system <system>
```
for both `linux` and `mac`. 

### Openslide
Note that for `openslide-python` to work, you need to install `openslide` separately on your system. 
See [here](https://openslide.org/download/) for instructions. 

### GDC client
To download the WSI data, you need to install the [gdc-client](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/) for your respective platform


## Data


### Download
From the root of the repository, run

1. Specify the path to the gdc-client executable in `main.yml` (this will likely be the repository root if you installed the dependencies using `invoke install`). 
2. Run `invoke download --dataset <dataset> --config_path <config>`, e.g., invoke download --dataset brca

If you are unsure about which arguments are available, you can always run `invoke download --help`.

The script downloads the data using the given manifest files in `data/tcga/gdc_manifests/full` and save it in the data folder under `tcga/wsi/<dataset>` taking the following structure: 

```
tcga/wsi/<dataset>/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

### Preprocessing

To ensure comparability with baselines, want to have the option to run the model in the WSI patches and extracted features using the [CLAM](https://github.com/mahmoodlab/CLAM) package. 

To extract he patches, run

```bash 
invoke preprocess --dataset <dataset> --config <config> --level <level>
```
Which will extract to the following structure

```
tcga/wsi/<dataset>_preprocessed/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```

Note that the slide.h5 files contain the coordinates of the patches that are to be read in 
via OpenSlide (x, y coordinates). 


## Running Experiments

### Single run

Given the configuration in `config.yml`, you can launch a single run using. Note that all below commands assume that you are in the repository root. 

```bash
python3 x_perceiver/main.py 
```

You can view the available command line arguments using 

```bash
python3 x_perceiver/main.py --help
```


### Hyperparameter search

You can launch a hyperparameter search by passing the `--hyperparameter_sweep` argument. 

```bash
python3 x_perceiver/main.py --hyperparameter_sweep
```

Note that the sweep parameters are specified in the `config/sweep.yaml` file. If a parameter is not specified as part of the parameter sweep, the program will default to whatever is configured in `config/main_gpu.yml`