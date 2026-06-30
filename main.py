"""
DeepU — minimal entry point.

Supported operations:
    1. --train_from_scratch     train the original model
    2. --retrain_from_scratch   retrain on the retain set (gold-standard reference)
    3. --method deepu           apply DeepU unlearning to a trained original model

Scope: datasets = cifar10 / cifar100, models = resnet18 / vgg16.

Examples
--------
# 1) Train original model
python main.py --train_from_scratch --dataset_name cifar10 --model_name resnet18

# 2) Retrain from scratch on the retain set (forgetting class index permutation_map[0])
python main.py --retrain_from_scratch --dataset_name cifar10 --model_name resnet18 --forget_class 1

# 3) Apply DeepU unlearning
python main.py --method deepu --dataset_name cifar10 --model_name resnet18 --forget_class 1
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm

from utils import *
from trainer import *
import deepu
import log_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeepU Machine Unlearning (cifar10: train / retrain / deepu)")

    # ----- method / dataset / model -----
    # Only "deepu" is a real unlearning method here; "pass" is a no-op used when you
    # just want to train_from_scratch / retrain_from_scratch.
    parser.add_argument('--method', type=str, default="deepu",
                        choices=["deepu", "pass"], help='unlearning method (only deepu supported)')
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset name (cifar10 or cifar100)')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        choices=['resnet18', 'vgg16'], help='model name (resnet18 or vgg16)')
    parser.add_argument('--exps_dir', type=str, default="./boundary_unlearn32/classification/exps",
                        help='experiments directory')

    # ----- DeepU-specific tuning -----
    parser.add_argument('--retune_deepu', action='store_true',
                        help='Retune and save to a specific folder (reuses cached mapping/)')
    parser.add_argument('--retune_folder', type=str, default=None,
                        help='Folder path to save retuned model (used with --retune_deepu)')
    parser.add_argument('--percentile_threshold_influential', type=int, default=75,
                        help='Percentile threshold for influential weights (deepu only). '
                             'Lower => more influential weights zeroed => stronger forgetting.')
    parser.add_argument('--percentile_threshold_shared', type=int, default=60,
                        help='Percentile threshold for shared weights (deepu only)')

    # ----- DeepU advanced tuning -----
    # NOTE: --snr_low/--snr_high and --grad_max_batches only take effect on a FRESH mapping
    # folder. If you re-tune them, delete the cached `.../deepu/mapping/` folder (or use a new
    # --description) first, otherwise the cached SNR mapping is reused.
    parser.add_argument('--snr_low', type=float, default=0.7,
                        help='SNR below this => Non-Influential weight (deepu only)')
    parser.add_argument('--snr_high', type=float, default=1.1,
                        help='SNR above this => Influential weight (deepu only). '
                             'Lower => more weights treated as influential => stronger forgetting.')
    parser.add_argument('--grad_max_batches', type=int, default=64,
                        help='Number of batches used to estimate per-weight gradients (deepu only)')
    parser.add_argument('--get_layers_first_n', type=int, default=20,
                        help='How many of the first layers to unlearn (deepu only)')
    parser.add_argument('--get_layers_last_n', type=int, default=15,
                        help='How many of the last layers to unlearn (deepu only)')
    parser.add_argument('--noise_scale', type=float, default=1.0,
                        help='Noise scale applied to shared weights during the KMeans update (deepu only)')
    parser.add_argument('--decay_factor', type=float, default=0.1,
                        help='Weight-decay factor applied to shared weights (deepu only)')
    parser.add_argument('--deepu_lr', type=float, default=0.001,
                        help='Learning rate for the retain backprop stage (deepu only)')
    parser.add_argument('--bp_every_n_layers', type=int, default=1,
                        help='Run a retain-backprop pass after updating this many layers (deepu only)')

    # ----- train / retrain from scratch -----
    parser.add_argument('--train_from_scratch', action='store_true', help='Train original model from scratch')
    parser.add_argument('--retrain_from_scratch', action='store_true', help='Retrain model from scratch (retain set only)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')

    # ----- lr / epoch settings (None => pulled from the dataset/model yaml config) -----
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--pretrain_epoch', type=int, default=None, help='train-from-scratch epochs')
    parser.add_argument('--pretrain_lr', type=float, default=None, help='pretrain learning rate')
    parser.add_argument('--unlearn_epoch', type=int, default=None, help='unlearning epochs')
    parser.add_argument('--unlearn_rate', type=float, default=None, help='unlearning learning rate')

    # ----- forget task -----
    parser.add_argument('--forget_class', type=int, nargs='+', default=[1],
                        help='Classes to forget. Single small value N (<=20) -> first N from permutation_map. '
                             'Multiple values or large index -> used directly as class indices.')

    # ----- misc -----
    parser.add_argument("--description", type=str, default="", help="Description suffix for this run")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--wo_dataaug", action="store_true")
    parser.add_argument('--load_original_model_path', type=str, default=None)
    parser.add_argument('--load_retrain_model_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root directory containing the CIFAR data.')
    parser.add_argument('--run_mia', action='store_true',
                        help='Run SVC-MIA membership inference after unlearning (off by default)')

    args = parser.parse_args()

    # Pull any unset hyperparameters from the dataset/model yaml config.
    config = OmegaConf.load(f'config/{args.dataset_name}_{args.model_name}.yaml')
    keys = ["pretrain_epoch", "pretrain_lr", "batch_size", "unlearn_epoch", "unlearn_rate"]
    for key in keys:
        if getattr(args, key) is None:
            setattr(args, key, config[key])
    if any(getattr(args, key) is None for key in keys):
        raise ValueError("some keys are not set")

    print(args)

    model_name = args.model_name
    num_workers = args.num_workers

    # ----- Resolve forget class indices + a short label for directory names -----
    # Single small value N (<=20) -> first N classes from permutation_map.
    # Multiple values (or a large index) -> used directly as class indices.
    _fc_arg = args.forget_class
    if len(_fc_arg) == 1 and _fc_arg[0] <= 20:
        forget_class = _fc_arg[0]
        _use_permmap = True
    else:
        forget_class = len(_fc_arg)
        _use_permmap = False

    description = f"{args.dataset_name}_{model_name}_forget{forget_class}"
    method_description = f"{args.method}_{args.description}" if args.description else f"{args.method}"

    seed_torch(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'only support cuda'

    path = Path(args.exps_dir).expanduser()
    create_dir(path)

    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name, wo_dataaug=args.wo_dataaug)

    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test,
                                    path=Path(args.data_dir).expanduser())
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, num_workers)

    num_classes = max(train_loader.dataset.targets) + 1

    permutation_map = getattr(config, "permutation_map")
    if _use_permmap:
        forget_class_index = permutation_map[:forget_class]
    else:
        forget_class_index = _fc_arg

    assert all(c < num_classes for c in forget_class_index), \
        f'All forget class indices must be < num_classes ({num_classes}). Got {forget_class_index}'
    note_print(f"forget class index: {forget_class_index}")

    num_forget = float("inf")
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
    train_forget_index, train_remain_index, test_forget_index, test_remain_index \
        = get_unlearn_loader(trainset, testset, forget_class_index, args.batch_size, num_forget, num_workers)

    # train/retrain checkpoints go to a separate folder so they never clobber the real ones.
    if args.train_from_scratch or args.retrain_from_scratch:
        ckpt_path = path / "test_pretrained_model"
        create_dir(ckpt_path)
    else:
        ckpt_path = path / "pretrained_model"

    formatted_time = datetime.now().strftime("%m%d-%H%M%S")

    # =====================================================================
    # 1) TRAIN ORIGINAL MODEL FROM SCRATCH
    # =====================================================================
    ori_model, retrain_model = None, None
    if args.train_from_scratch:
        print('=' * 100)
        print(' ' * 25 + 'train original model from scratch')
        print('=' * 100)
        ori_model = train_save_model(train_loader, test_loader, model_name, args.optim_name, args.pretrain_lr,
                                     args.pretrain_epoch, ckpt_path,
                                     f"{args.dataset_name}_{model_name}_original_model_{args.description}_{formatted_time}")
        print('\noriginal model acc:\n', test_each_classes(ori_model, test_loader, num_classes))

    # =====================================================================
    # 2) RETRAIN FROM SCRATCH (retain set only — gold-standard reference)
    # =====================================================================
    if args.retrain_from_scratch:
        print('=' * 100)
        print(' ' * 25 + 'retrain model from scratch')
        print('=' * 100)
        retrain_model = train_save_model(train_remain_loader, test_remain_loader, model_name, args.optim_name,
                                         args.pretrain_lr, args.pretrain_epoch, ckpt_path,
                                         f"{args.dataset_name}_{model_name}_retrain_forget{forget_class}_model_{args.description}_{formatted_time}")
        per_class_str = test_each_classes(retrain_model, test_loader, num_classes)
        print('\nretrain model per-class acc:\n', per_class_str)
        lines = [l for l in per_class_str.strip().split('\n') if l]
        forget_accs = [lines[i] for i in forget_class_index if i < len(lines)]
        print(f"\n[Retrain summary]  Forget class indices: {forget_class_index}")
        print(f"  Forget class acc  (should be ~0%): {', '.join(forget_accs)}")
        _, retain_acc = test(retrain_model, test_remain_loader)
        print(f"  Retain test acc   (should be high): {retain_acc:.2%}")

    if args.train_from_scratch or args.retrain_from_scratch:
        note_print('train/retrain from scratch done, exiting')
        exit(0)

    # =====================================================================
    # 3) APPLY DEEPU  (load original + retrain reference, then unlearn)
    # =====================================================================
    print('=' * 100)
    print(' ' * 25 + 'load original model and retrain model')
    print('=' * 100)

    if args.load_original_model_path:
        original_model_path = Path(args.load_original_model_path)
    else:
        original_model_path = ckpt_path / f'{args.dataset_name}_{model_name}_original_model.pth'
    note_print(f"load original model from {original_model_path}")
    ori_model = load_model(original_model_path, model_name, num_classes)

    if not args.debug:
        _, acc = test(ori_model, train_forget_loader);   print(f"forget train acc:{acc:.2%}")
        _, acc = test(ori_model, train_remain_loader);   print(f"remain train acc:{acc:.2%}")
        _, acc = test(ori_model, test_forget_loader);    print(f"forget test acc:{acc:.2%}")
        _, acc = test(ori_model, test_remain_loader);    print(f"remain test acc:{acc:.2%}")

    if args.load_retrain_model_path:
        retrain_model_path = Path(args.load_retrain_model_path)
    else:
        retrain_model_path = ckpt_path / f'{args.dataset_name}_{model_name}_retrain_forget{forget_class}_model.pth'
    note_print(f"load retrain model from {retrain_model_path}")
    # The retrain model is an optional reference; load it if present.
    retrain_model = load_model(retrain_model_path, model_name, num_classes)

    if not args.debug:
        note_print("\nretrain model performance:")
        _, acc = test(retrain_model, train_forget_loader);  print(f"forget train acc:{acc:.2%}")
        _, acc = test(retrain_model, train_remain_loader);  print(f"remain train acc:{acc:.2%}")
        _, acc = test(retrain_model, test_forget_loader);   print(f"forget test acc:{acc:.2%}")
        _, acc = test(retrain_model, test_remain_loader);   print(f"remain test acc:{acc:.2%}")

    retune = bool(getattr(args, "retune_deepu", False))
    retune_dir = getattr(args, "retune_folder", None)
    if retune and retune_dir:
        experiment_dir = Path(retune_dir).expanduser()
    else:
        experiment_dir = path / description / method_description
        create_dir(path / description)
        create_dir(path / description / method_description)
    create_dir(experiment_dir)

    OmegaConf.save(OmegaConf.create(vars(args)), experiment_dir / "config.yaml")
    logger, console_handler = log_utils.setup_logger(experiment_dir, logger_name="train_log")
    log_utils.enable_console_logging(logger, console_handler, True)

    loader_dict = {"train_forget": train_forget_loader, "train_remain": train_remain_loader,
                   "test_forget": test_forget_loader, "test_remain": test_remain_loader,
                   "test": test_loader}

    print('*' * 100)
    note_print(' ' * 25 + f'begin {args.method.replace("_", " ")} unlearning')
    print('*' * 100)

    experiment_path = path / description / method_description
    unlearn_model = None

    if args.method == "deepu":
        layers_to_update = deepu.get_layers(ori_model, first_n=args.get_layers_first_n,
                                            last_n=args.get_layers_last_n)

        if retune and retune_dir:
            mapping_folder = Path(args.retune_folder) / "mapping"
        else:
            mapping_folder = path / description / method_description / "mapping"
        mapping_folder.mkdir(parents=True, exist_ok=True)

        # Stage 1+2: gradient-influence mapping + KMeans weight update (per layer, cached).
        for layer_name in tqdm(layers_to_update, desc="Updating layers"):
            print(f"Processing layer: {layer_name}")
            _ = deepu.compute_grad_influence_mapping(
                ori_model, train_forget_loader, train_remain_loader,
                first_layer_name=layer_name, exp_dir=mapping_folder, device=device,
                max_batches=args.grad_max_batches, snr_low=args.snr_low, snr_high=args.snr_high)
            _ = deepu.update_weights_with_kmeans(
                layer_name, mapping_folder,
                percentile_threshold_influential=args.percentile_threshold_influential,
                percentile_threshold_shared=args.percentile_threshold_shared,
                noise_scale=args.noise_scale, decay_factor=args.decay_factor)

        # Stage 3: apply updated weights + retain backprop.
        unlearn_model = deepu.update_model(
            model=ori_model, layers_list=layers_to_update, mapping_folder=mapping_folder,
            retain_loader=train_remain_loader, forget_loader=train_forget_loader,
            test_loader=test_loader, device=device,
            learning_rate=args.deepu_lr, bp_every_n_layers=args.bp_every_n_layers)

    elif args.method == "pass":
        pass
    else:
        raise ValueError('method not found')

    # ----- Save the unlearned checkpoint -----
    if unlearn_model:
        if retune and retune_dir:
            save_path = Path(args.retune_folder) / "ckpt.pth"
        else:
            save_path = path / description / method_description / "ckpt.pth"
        torch.save(unlearn_model, save_path)
        note_print(f"saved unlearned model to {save_path}")

        # ----- Accuracy metrics for the unlearned model -----
        _, forget_train_acc = test(unlearn_model, train_forget_loader)
        _, remain_train_acc = test(unlearn_model, train_remain_loader)
        _, forget_test_acc = test(unlearn_model, test_forget_loader)
        _, remain_test_acc = test(unlearn_model, test_remain_loader)
        print(f"forget train acc: {forget_train_acc:.2%}")
        print(f"remain train acc: {remain_train_acc:.2%}")
        print(f"forget test acc: {forget_test_acc:.2%}")
        print(f"remain test acc: {remain_test_acc:.2%}")

    if args.run_mia and unlearn_model:
        import evaluation
        import random
        test_remain_len = len(test_remain_index)
        random.shuffle(train_remain_index)
        train_remain_index = train_remain_index[:test_remain_len]
        logger.info(f"train remain size: {len(train_remain_index)}")
        train_remain_sampler = SubsetRandomSampler(train_remain_index)
        mia_train_remain_loader = DataLoader(train_remain_loader.dataset, batch_size=args.batch_size,
                                             sampler=train_remain_sampler)
        logger.info("start mia evaluation")
        mia_result = evaluation.SVC_MIA(
            shadow_train=mia_train_remain_loader,
            shadow_test=test_remain_loader,
            target_train=train_forget_loader,
            target_test=None,
            model=unlearn_model,
        )

        # Print the SVC-MIA result to the console so it is visible at the end of the run.
        # We report the confidence-based score as the MIA score.
        print('=' * 100)
        note_print(f"MIA score: {mia_result['confidence']:.4f}")
        print('=' * 100)

    logger.info("done")
    exit()
