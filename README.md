# DeepU: Machine Unlearning

> **Status:** Initial release: this repository currently includes the core DeepU method, which is fully functional and runnable end to end. Remaining components from the paper will be added in upcoming updates.

DeepU is a class-unlearning method for image classifiers. Given a model trained on the full dataset, it removes the influence of one or more target ("forget") classes while preserving accuracy on the remaining ("retain") classes.

<!-- IMAGE 1: Overview figure, e.g. Fig. 2 (targeted weight update strategy) from the paper. Suggested path: assets/deepu_overview.png Add here as: ![DeepU overview](assets/deepu_overview.png) -->

For our work we used: [Delete](https://github.com/shaaaaron/DELETE), [Model Inversion Attack](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox), and [If-GMI](https://github.com/final-solution/IF-GMI) repositories. We thank all the researchers for their contributions. 


This repository contains a self-contained implementation with three operations:

1. **`--train_from_scratch`**: train the original model on the full dataset.
2. **`--retrain_from_scratch`**: retrain on the retain set only (the gold-standard reference for what a perfectly unlearned model should look like).
3. **`--method deepu`**: apply DeepU unlearning to an already-trained original model.

**Scope:** experiments were run on **CIFAR-10** and **CIFAR-100** with **ResNet-18** and **VGG-16**. Both datasets are downloaded automatically on first run.

## Layout
```
deepu/
├── main.py          # entry point: train / retrain / deepu
├── deepu.py         # the DeepU method (gradient-influence mapping + KMeans + retain backprop)
├── trainer.py       # train / test / train_save_model / test_each_classes
├── utils.py         # datasets, transforms, unlearn loaders, seeding
├── log_utils.py     # logger setup
├── models/          # resnet.py + vgg.py + get_model / load_model
├── config/           # <dataset>_<model>.yaml hyperparameters
└── evaluation/       # SVC-MIA membership inference (optional, via --run_mia)
```

## Requirements
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install omegaconf scikit-learn pandas tqdm matplotlib
```
A CUDA GPU is required. The dataset is downloaded automatically to `--data_dir` (default `./data`).

## Usage

### 1) Train the original model
```bash
python main.py --train_from_scratch --dataset_name cifar10  --model_name resnet18
python main.py --train_from_scratch --dataset_name cifar100 --model_name vgg16
```

### 2) Retrain from scratch (retain set only)
```bash
python main.py --retrain_from_scratch --dataset_name cifar10 --model_name resnet18 --forget_class 1
```

### 3) Apply DeepU unlearning
```bash
python main.py --method deepu --dataset_name cifar10 --model_name resnet18 --forget_class 1
```

### Selecting the forget classes
`--forget_class` controls which classes are removed:
- A single small value `N` (≤ 20) forgets the **first `N`** classes listed in the config's `permutation_map` (e.g. `--forget_class 1` forgets `permutation_map[0]`).
- Multiple values, or a single large index, are used **directly as class indices** (e.g. `--forget_class 4 3 5` forgets exactly those classes).

Other useful flags:
- On Windows, pass `--num_workers 0`.
- Add `--run_mia` to run SVC-MIA membership inference after unlearning.

### Checkpoints
- `--train_from_scratch` / `--retrain_from_scratch` save under `<exps_dir>/test_pretrained_model/`.
- DeepU loads the original model from `<exps_dir>/pretrained_model/` by default (override with `--load_original_model_path`). The unlearned model is saved to `<exps_dir>/<dataset>_<model>_forget<N>/deepu/ckpt.pth`.

## How DeepU works
1. **Gradient-influence mapping**: for each layer, compare how each weight responds to the forget data vs. the retain data (signal-to-noise ratio). Weights are tagged `Non_Influential` / `Shared` / `Influential`. Results are cached under `.../deepu/mapping/`.
2. **KMeans weight update**: influential weights beyond a distance percentile are zeroed; shared weights receive weight decay.
3. **Retain backpropagation**: fine-tune the modified model on the retain data to recover accuracy on the classes that should be kept.


