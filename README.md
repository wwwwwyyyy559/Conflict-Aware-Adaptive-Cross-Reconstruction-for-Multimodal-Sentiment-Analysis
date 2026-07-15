# CACR-CVPR2026
The code for Conflict-Aware Adaptive Cross-Reconstruction for Multimodal Sentiment Analysis, which is accepted by CVPR 2026.

## Framework

![Framework of CACR](framework.jpg)

**Figure 1.** Overview of the proposed CACR framework. CACR decomposes each modality into shared and modality-specific representations, estimates cross-modal conflicts, and performs adaptive cross-reconstruction. The reconstructed shared information and enhanced modality-specific information are subsequently integrated for multimodal sentiment prediction.


## Datasets

Experiments are conducted on the following public multimodal sentiment analysis datasets:

* [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
* [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)

The processed feature files for CMU-MOSI and CMU-MOSEI can be downloaded from:

[Download the processed datasets](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing)

After downloading the datasets, place them in the `./dataset` directory. 

The dataset paths should be specified in `./config/config.json`. For example, when the processed CMU-MOSI feature file is located at:

```text
./dataset/MOSI/aligned_50.pkl
```

the corresponding configuration should be:

```json
{
  "dataset_root_dir": "./dataset",
  "featurePath": "MOSI/aligned_50.pkl"
}
```

Please revise the dataset path according to your local directory structure.

### Data Notice

The provided files contain processed multimodal features rather than the original videos or raw modality data. The metadata and raw data are not redistributed because of the privacy and copyright considerations associated with YouTube content creators.

For access to the original datasets and further details about dataset processing, please refer to the official [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK).

## Requirements

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
Python 3.9.13
PyTorch 2.0.0
CUDA 12.0
```

The exact package versions used in the experiments should be listed in `requirements.txt`.

## Training

Before training, specify the dataset in `train.py` as either:

```python
dataset_name = "mosi"
```

or:

```python
dataset_name = "mosei"
```

Then run:

```
python3 train.py
```

By default, the trained model checkpoints are saved in:

```
./pt
```


## Citation

Please cite our paper if you find this repository or the CACR method useful in your research:

```bibtex
@InProceedings{Wang_2026_CVPR,
    author    = {Wang, Yan and Cao, Fuyuan and Zhao, Xingwang},
    title     = {Conflict-Aware Adaptive Cross-Reconstruction for Multimodal Sentiment Analysis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {15880--15889}
}
```
