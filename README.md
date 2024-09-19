# UniAMP
The code, data and models in the paper **UniAMP: Enhancing AMP Prediction using Deep Neural Networks with Inferred Information of Peptides**\
UniAMP can be accessed online via [https://amp.starhelix.cn](https://amp.starhelix.cn)

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
pip install sentencepiece
pip install transformers
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
### Feature extraction
**The UniRep repository**, available at [https://github.com/churchlab/UniRep](https://github.com/churchlab/UniRep), contains the source code, documentation, and examples for using UniRep in your projects.\
Due to the installation of the `tape_proteins` toolkit, you can conveniently utilize UniRep as follows:
~~~
tape-embed unirep input.fasta output.npz babbler-1900 --tokenizer unirep
~~~
**The ESM-2 repository**, available at [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm).\
When you need to use ESM-2 feature vectors, you can use the `utils/cal_esm2.py` program to convert the `.fasta` format file to the `.json` format, and then use the `utils/data.py/load_esm2` function to load it.\
**The ProtTrans repository**, available at [https://github.com/agemagician/ProtTrans](https://github.com/agemagician/ProtTrans).\
When you need to use ProtT5 feature vectors, you can use the `utils/cal_protT5.py` program to convert the `.fasta` format file to the `.json` format, and then use the `utils/data.py/load_protT5` function to load it.\
Before using it, you must download all files from [https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main](https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main) and put them in the `utils/protT5` directory

### train
~~~
python train.py -model UniAMP
~~~
_If you need to train other models, the `-dataset_path` should be `*.csv`, and `-feature pca` for feature PCA._
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
### test
~~~
python test.py -model UniAMP -model_path ./models/model_aeruginosa_UniAMP_uni_protT5.h5
~~~
_If you need to test other models, the `-dataset_path` should be `*.csv`._
### infer
~~~
python infer.py -model UniAMP -model_path ./models/model_aeruginosa_UniAMP_uni_protT5.h5 -dataset_path ./data_aeruginosa/benchmark_dataset.fasta
~~~
- Use `-save_path` to specify the path where the inference results will be saved.
  - Default Value: _./data/results/results.csv_
## Data Format
- When inferring, there are no special requirements for the `.fasta` format.
- When training and test:
  - A DataFrame is recorded in `.csv` format, which is required to contain `sequence` and `class` columns. Where `sequence` should be composed of 20 standard amino acids, and `class` is the label of the sequence.
  - When using UniRep, ESM-2 and protT5 to convert `.fasta` to `.npz/.json` format, please ensure that the last character of your `sequence_name` is the label of the sequence. The program will read this character for evaluation.
