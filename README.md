# UniAMP
The code and data in the paper **UniAMP: Enhancing AMP Prediction using Deep Neural Networks with Inferred Information of Peptides**

## Requirements
To set up the necessary environment for UniAMP, follow these steps:
1. **Create and Activate Conda Environment**
~~~
conda create -n UniAMP python=3.7
conda activate UniAMP
~~~
2. **Install Python Packages**
~~~
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorflow-gpu==1.14.0
pip install keras==2.2.4
pip install scikit-learn==0.22.1
pip install pandas tape_proteins pytorch_pretrained_bert
~~~
**You may need the following code when protobuf error occurs**
~~~
pip install protobuf==3.20.0
~~~

**When encountering model loading errors, please verify the correctness of your H5py installation.**
~~~
pip install h5py==2.9.0
~~~
Ensure to adjust the version numbers and dependencies according to your specific requirements. This setup assumes the use of a Conda environment named 'UniAMP'.

## Usage
###Feature extraction
In this project, we leverage the power of [UniRep](https://github.com/churchlab/UniRep), a tool developed by the [Church Lab](https://churchlab.github.io/), for feature extraction. UniRep is a versatile toolkit for extracting fixed-size feature vectors from biological sequences, such as proteins or RNA.\
**The UniRep repository**, available at [https://github.com/churchlab/UniRep](https://github.com/churchlab/UniRep), contains the source code, documentation, and examples for using UniRep in your projects.\
Due to the installation of the `tape_proteins` toolkit, you can conveniently utilize UniRep as follows:
~~~
tape-embed unirep input.fasta output.npz babbler-1900 --tokenizer unirep
~~~
###train
~~~
python train.py -model UniAMP -dataset_path ./data/aeruginosa/training_dataset.npz
~~~
_If you need to train other models, your `dataset_path` should be `*.csv`, and `-feature pca` for feature PCA._
- Use `-lr xxx` to set a specified learning rate.
  - Default Value: _1e-4_
- Use `-epochs xxx` to set the number of training epochs.
  - Default Value: _200_
- Use `-weight` to determine whether to use a weighted criterion.
  - Default Value: _False_
- Use `-random_seed xxx` to set the random seed.
- Use `-save_dir xxx` and `-log_dir xxx` to respectively set the directories for saving models and logs.
  - Default Value: _./models_ and _./logs_
- Use `-train_batch_size xxx` and `-test_batch_size xxx` to set the training and testing batch sizes.
  - Default Value: _256_ and _256_
- Use `-val_pro xxx` to set the proportion of the validation set.
  - Default Value: _0.2_
- Use `-patience xxx` to set the patience for training.
  - Default Value: _30_
###test
~~~
python test.py -model UniAMP -model_path ./models/model_UniAMP_uni.h5 -dataset_path ./data_aeruginosa/benchmark_dataset.npz
~~~
_If you need to test other models, your `dataset_path` should be `*.csv`._
###infer
~~~
python infer.py -model UniAMP -model_path ./models/model_UniAMP_uni.h5 -dataset_path ./data_aeruginosa/benchmark_dataset.fasta
~~~
- Use `-save_path` to specify the path where the inference results will be saved.
  - Default Value: _./data/results/results.csv_


