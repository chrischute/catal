# Çatalhöyük

![West Mound at Çatalhöyük](/img/west-mound.jpg?raw=true "West Mound at Çatalhöyük")

## Project Overview

[Çatalhöyük](https://en.wikipedia.org/wiki/%C3%87atalh%C3%B6y%C3%BCk) is an
UNESCO World Heritage Site located in modern day Turkey. This site has been
the subject of the Çatalhöyük Research Project, in which archaeologists have
collected over 150,000 photographs richly detailing buildings, artifacts,
and ways of life of this civilization.

As described on the [project's website](http://www.catalhoyuk.com/project), "Çatalhöyük has been the subject of investigation for more than 50 years. Researchers from around the world have travelled to the site over the past half-century to study its vast landscape of buildings, remarkable ways of life, and its many exquisite works of art and craft. Since 1993, the Çatalhöyük Research Project has recruited an international group of specialists to pioneer new archaeological, conservation and curatorial methods on and off site. Simultaneously, it aims to advance our understandings of human life in the past."

Our goal is to use computer vision to help enrich the digital resources
produced by the Çatalhöyük Research Project, in keeping with the project's
mission to create a living archive.

![Bull's Head](/img/bulls-head.jpg?raw=true "Bull's Head Uncovered at Çatalhöyük")

## Subprojects

### Whiteboards
A large number of photos of the dig site contain whiteboards with text describing
the photo's contents, and often the contents of subsequent photos. Our first project
is to detect photos with whiteboards, then extract the text from these
whiteboards and use optical character recognition to digitize the information.

### Metadata
Many of the photos in the Çatalhöyük database are already labeled with metadata.
To make the database more readily searchable, we aim to expand these tags as
thoroughly as possible to entire archive.

## Usage

### Creating Conda environment
  1. Make sure you have [Anaconda or Miniconda](https://conda.io/docs/download.html)
  2. Download repo and `cd` into it. Run `conda env create --file=environment.yml`.
  3. Add CUDA to the environment if you have a GPU.
  4. (Optional) Add interpreter to PyCharm
    - Go to `Preferences > Project > Project Interpreter`
    - Click the gear icon, then 'Add'
    - Select 'Conda Environment' and 'Existing Environment'
    - Click the three dots, then find the interpreter. Should be somewhere like
    `/Users/christopherchute/anaconda3/envs/catalhoyuk/bin/python`.

### Train
  1. Create and activate conda environment as described above.
  2. Run `python train.py --name=NAME --batch_size=128 --data_dir=data/wb130k`.

### Predict
  1. Locate your trained model in `ckpts` folder.
  2. Run `python predict.py --ckpt_path=ckpts/NAME/best.pth.tar --data_dir=data/wb130k/ --phase=test --name=wb130k_test --gpu_ids=0,1,2,3 --batch_size=256 --prob_threshold=0.4`.

### TensorBoard
  1. `cd` into this project's root directory.
  2. Run `tmux new -s tb` for  a new `tmux` session named `tb`.
  3. Run `source activate res` to set up the virtual environment.
  4. Run `tensorboard --logdir=. --port=5678`.
  5. Hit `ctrl-b`, `d` to detach from the `tmux` session. Later run `tmux a -t train` to re-attach.
  6. (Local) Run `ssh -N -f -L localhost:1234:localhost:5678 <remote_host>`.
  7. (Local) In a web browser, go to `http://localhost:1234`.

## Contents

```text
+ args: Command-line arg parsing
+ ckpts: Holds model checkpoints
+ data: Placeholder for CIFAR dataset
+ data_loader: Wraps CIFAR data loader
+ logger: Logs training info to the console and TensorBoard
+ logs: Holds logs produced by the logger
+ optim: Optimizer and learning rate scheduler
+ saver: Saves and loads model checkpoints
+ scripts: Scripts for miscellaneous tasks
+ util: utility functions

- train.py: Training
- test.py: Test inference
```
