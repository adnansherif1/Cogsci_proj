# Run Details.

This code is modified from the codebase linked with the paper to make it applicable to the questions answered in the final report.



## Installation

Setup your environment, e.g. using

```
conda install pyyaml
conda install pytorch torchvision -c pytorch
```

Optional, but recommended: Install `tqdm` top geht progress bars while running:

```
conda install tqdm
```

Download/Clone the code using

```
git clone https://github.com/adnansherif1/Cogsci_proj.git
cd PyTorch-AutoNEB
```

## Usage

### Running the examples

To evaluate the results simply run the **evaluation.ipynb** notebook it contains all the necessary details.

The yaml file used for project training is in **configs/cogsci.yaml**
The evaluation notebook is self explanatory.
The project code uses a **custom pytorch dataset**  that creates random pairing from a **1*2 input space to a 10*1** output space.
The dataset contains **30 examples**. A MLP with **2 hidden layers containing 10 units** each was used for training.
6 random initial minimas were used and the low energy path connecting them was found using the above setup.




## Results

The final MSTs for analysis with [Evaluate.ipynb](Evaluate.ipynb) can be found at [this repository](https://drive.google.com/drive/folders/1VQvecH3lWntBD5H0VHyYU577DunvkQlb).
As of now, it contains only a subset of systems. [Open an issue](https://github.com/fdraxler/PyTorch-AutoNEB/issues/new) to request more systems.
