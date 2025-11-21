## herpeton
<div style="display: flex; align-items: center; justify-content: space-between;">
  <div style="flex: 1;">
    <img src="images/herpeton_logo.png" alt="Herpeton Logo" width="150">
  </div>
  <div style="flex: 2; text-align: right;">
    <p><strong>herpeton</strong> is a computer vision project focused on reptile detection, conservation, and ecological monitoring using deep learning techniques.</p>
  </div>
</div>
The Greek word herpeton (ἑρπετόν) means “creeping thing” or “reptile.”

It comes from the Greek verb herpein (ἕρπειν), which means “to creep” or “to crawl.”
The term reflects how these animals, such as snakes, lizards, and other reptiles, move close to the ground.

The modern word herpetology literally means “the study of creeping animals.”

# Automated Reptile Species Classification Using the BioTrove Dataset


## AAI-521: Applied Computer Vision for AI  

**Project Status:** Active (In Progress)

# Project Introduction / Objective / Description

This project leverages the **BioTrove-TRAIN Reptilia subset** from the **BioTrove dataset** (Hugging Face), containing ~1.3 million labeled reptile images across 189 species.  
Images range from 224–1024 px and include taxonomy metadata (class, family, genus, species, scientific name, location).

The project develops a computer vision pipeline to identify **reptile species** such as snakes, lizards, turtles, and geckos from field imagery. Using the **BioTrove biodiversity dataset**, it explores how **CNNs**, **Vision Transformers (ViT)**, and **YOLOv10** can enhance conservation and biodiversity monitoring.



# Methods Used
- Deep Learning (CNNs, Vision Transformers)  
- Object Detection (YOLOv10 / Ultralytics)  
- Transfer Learning (ImageNet & BioTrove-CLIP)  
- Data Augmentation and Preprocessing  
- Model Evaluation & Visualization (Grad-CAM, Confusion Matrices)  
- Ethics for AI and Responsible Wildlife Applications  

# Technologies
- Python
- Google Colab / Jupyter Notebook  
- Hugging Face Datasets
- PyTorch / Ultralytics YOLOv10 
- NumPy, Pandas, Matplotlib, Seaborn
- ImageHash
- OpenCV, PIL
- TensorBoard

## Dataset Information

### BioTrove-TRAIN Reptilia Subset
- **Source**: [BGLab/BioTrove-Train](https://huggingface.co/datasets/BGLab/BioTrove-Train) on Hugging Face
- **Size**: ~1.3M reptile images from ~40M total biodiversity images
- **Species Coverage**: 189+ reptile species across major families
- **Image Resolution**: 224-1024 pixels
- **Format**: RGB images with comprehensive metadata

### Key Features
- **Taxonomic Hierarchy**: Full classification (Kingdom → Species)
- **Geographic Distribution**: Global coverage with location metadata
- **Species Diversity**: Snakes, lizards, turtles, geckos, and more
- **High Quality**: Curated from iNaturalist with expert validation
- **Metadata Rich**: Scientific names, common names, family information

### Major Reptile Families Included
- **Viperidae** (Vipers and Pit Vipers)
- **Colubridae** (Colubrids - largest snake family)
- **Elapidae** (Venomous snakes including cobras, mambas)
- **Gekkonidae** (Geckos)
- **Iguanidae** (Iguanas and related lizards)
- **Testudinidae** (Tortoises)
- **Cheloniidae** (Sea turtles)
- And many more...

### Usage in Computer Vision
This dataset is specifically designed for:
- **Species Classification**: Multi-class reptile identification
- **Hierarchical Classification**: Family/genus/species level predictions
- **Transfer Learning**: Pre-trained on biodiversity data
- **Object Detection**: Bounding box annotations available
- **Conservation Applications**: Real-world wildlife monitoring

<br>

# Installation

### Prerequisites: 

#### Python Version
For this project we are using Python version 3.12.2, conda automatically will install and set the correct python version for the project so there is nothing that needs to be done.

#### 1. Install Miniconda

If you are already using Anaconda or any other conda distribution, feel free to skip this step.

Miniconda is a minimal installer for `conda`, which we will use for managing environments and dependencies in this project. Follow these steps to install Miniconda or go [here](https://docs.anaconda.com/miniconda/install/) to reference the documentation: 

1. Open your terminal and run the following commands:
```bash
   $ mkdir -p ~/miniconda3

   <!-- If using Apple Silicon chip M1/M2/M3 -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
   <!-- If using intel chip -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh

   $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   $ rm ~/miniconda3/miniconda.sh
```

2. After installing and removing the installer, refresh your terminal by either closing and reopening or running the following command.
```bash
$ source ~/miniconda3/bin/activate
```

3. Initialize conda on all available shells.
```bash
$ conda init --all
```

You know conda is installed and working if you see (base) in your terminal. Next, we want to actually use the correct environments and packages.

#### 2. Install Make

Make is a build automation tool that executes commands defined in a Makefile to streamline tasks like compiling code, setting up environments, and running scripts. [more information here](https://formulae.brew.sh/formula/make)

##### Installation

`make` is often pre-installed on Unix-based systems (macOS and Linux). To check if it's installed, open a terminal and type:
```bash
make -v
```

If it is not installed, simply use brew:
```bash
$ brew install make
```


#### Step-by-step Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd herpeton

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install required packages
pip install -r requirements.txt

# Start with the dataset
python quick_start.py
```

## Quick Setup

1. **Install dependencies:**
   ```bash
  $ make create
        or
  $ make update
   ```

2. **Quick start with the dataset:**
   ```bash
   make quick_start
   ```

3. **Or use the Jupyter notebook for detailed exploration:**
   ```bash
   make notebook biotrove_reptilia_loader.ipynb
   ```


#### Available Commands

The following commands are available in this project’s `Makefile`:

- **Set up the environment**:

    This will create the environment from the environment.yml file in the root directory of the project.

    ```bash
      $ make create
    ```

- **Update the environment**:

    This will update the environment from the environment.yml file in the root directory of the project. Useful if pulling in new changes that have updated the environment.yml file.

    ```bash
      $ make update
    ```

- **Remove the environment**:

    This will remove the environment from your shell. You will need to recreate and reinstall the environment with the setup command above.

    ```bash
    $ make clean
    ```

- **Activate the environment**:

    This will activate the environment in your shell. Keep in mind that make will not be able to actually activate the environment, this command will just tell you what conda command you need to run in order to start the environment.

    Please make sure to activate the environment before you start any development, we want to ensure that all packages that we use are the same for each of us.

    ```bash
    $ make activate
    ```

    Command you actually need to run in your terminal:
    ```bash
    $ conda activate herpeton
    ```

- **Deactivate the environment**:

    This will Deactivate the environment in your shell.

    ```bash
    $ make deactivate
    ```

- **Quick start**:

    This command will run the quick_start python script to generate the dataset

    ```bash
    $ make quick
    ```

- **run jupyter notebook**:

    This command will run jupyter notebook from within the conda environment. This is important so that we can make sure the package versions are the same for all of us! Please make sure that you have activated your environment before you run the notebook.

    ```bash
    $ make notebook
    ```

- **Export packages to env file**:

    This command will export any packages you install with either `conda install ` or `pip install` to the environment.yml file. This is important because if you add any packages we want to make sure that everyones machine knows to install it.

    ```bash
    $ make freeze
    ```

- **Verify conda environment**:

    This command will list all of your conda envs, the environment with the asterick next to it is the currently activated one. Ensure it is correct.

    ```bash
    $ make verify
    ```

#### Example workflows:

To simplify knowing which commands you need to run and when you can follow these instructions:

- **First time running, no env installed**:

    In the scenario where you just cloned this repo, or this is your first time using conda. These are the commands you will run to set up your environment.

    ```bash
    <!-- Make sure that conda is initialized -->
    $ conda init --all

    <!-- Next create the env from the env file in the root directory. -->
    $ make create

    <!-- After the environment was successfully created, activate the environment. -->
    $ conda activate herpeton

    <!-- verify the conda environment -->
    $ make verify

    <!-- verify the python version you are using. This should automatically be updated to the correct version 3.12.2 when you enter the environment. -->
    $ python --version

    <!-- Run jupyter notebook and have some fun! -->
    $ make notebook
    ```

- **Installing a new package**:

    While we are developing, we are going to need to install certain packages that we can utilize. Here is a sample workflow for installing packages. The first thing we do is verify the conda environment we are in to ensure that only the required packages get saved to the environment. We do not want to save all of the python packages that are saved onto our system to the `environment.yml` file. 

    Another thing to note is that if the package is not found in the conda distribution of packages you will get a `PackagesNotFoundError`. This is okay, just use pip instead of conda to install that specific package. Conda thankfully adds them to the environment properly.

    ```bash
    <!-- verify the conda environment -->
    $ make verify

    <!-- Install the package using conda -->
    $ conda install <package_name>

    <!-- If the package is not found in the conda channels, install the package with pip. -->
    $ pip install <package_name>

    <!-- If removing a package. -->
    $ conda remove <package_name>
    $ pip remove <package_name>

    <!-- Export the package names and versions that you downloaded to the environment.yml file -->
    make freeze
    ```

- **Daily commands to run before starting development**:

    Here is a sample workflow for the commands to run before starting development on any given day. We want to first pull all the changes from github into our local repository, 

    ```brew
    <!-- Pull changes from git -->
    $ git pull origin main

    <!-- Update env based off of the env file. It is best to deactivate the conda env before you do this step-->
    $ conda deactivate
    $ make update
    $ conda activate herpeton

    $ make notebook
    ```

- **Daily commands to run after finishing development**:

    Here is a sample workflow for the commands to run after finishing development for any given day.

    ```brew
    $ conda deactivate

    <!-- If you updated any of the existing packages, freeze to the environment.yml file. -->
    $ make freeze

    <!-- Commit changes to git -->
    $ git add .
    $ git commit -m "This is my commit message!"
    $ git push origin <branch_name>
    ```


## Contributors
<table>
  <tr>
    <td>
        <a href="https://github.com/littlecl42.png">
          <img src="https://github.com/littlecl42.png" width="100" height="100" alt="Carrie Little"/><br />
          <sub><b>Carrie Little</b></sub>
        </a>
      </td>
      <td>
        <a href="https://github.com/mojodean.png">
          <img src="https://github.com/mojodean.png" width="100" height="100" alt="Dean P. Simmer"/><br />
          <sub><b>Dean P. Simmer </b></sub>
        </a>
      </td>
     <td>
      <a href="https://github.com/omarsagoo.png">
        <img src="https://github.com/omarsagoo.png" width="100" height="100" alt="Omar Sagoo"/><br />
        <sub><b>Omar Sagoo</b></sub>
      </a>
    </td>
  </tr>
</table>