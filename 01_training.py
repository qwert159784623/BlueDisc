import argparse
import os

import mlflow.pytorch
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
from torch.utils.data import DataLoader
from module.discriminator import DBuilder
from module.gan_model import GANModel
from module.generator import GBuilder
from module.logger import MLFlowLogger
from module.device_manager import DeviceManager
from module.pipeline import AugmentationsBuilder
from module.random_seed import RandomSeedManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True, choices=["D", "N"], help="Label type to train the generator on (D for detection, N for noise)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name available in seisbench dataset class name (e.g., ETHZ, InstanceCount)")
    parser.add_argument("--g-lr", type=float, help="Generator learning rate")
    parser.add_argument("--d-lr", type=float, help="Discriminator learning rate")
    parser.add_argument("--data-weight", type=float, help="Data loss weight. If specified, GAN will be used.")
    parser.add_argument("--sample-size", type=int, default=1, help="Number of samples to use from dev set for evaluation during training")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g., 'cpu', 'cuda', 'auto')")

    args = parser.parse_args()

    seed_value = 42
    seed_manager = RandomSeedManager(seed_value)
    seed_manager.set_seed()

    # Initialize device manager
    device_manager = DeviceManager(args.device)

    sample_size = args.sample_size
    batch_size = args.batch_size
    max_steps = args.max_steps

    # Determine if GAN is used based on whether data_weight is specified
    use_gan = args.data_weight is not None

    if not use_gan:
        # No GAN, only data loss
        gan_loss_weight = 0
        data_weight = 1.0
        gan_type = None
    else:
        # Use GAN with specified data_weight
        gan_loss_weight = 1.0
        data_weight = args.data_weight
        gan_type = "SGAN"

    # Build generator model (only PN/PhaseNet)
    g_builder = GBuilder()
    g_model = g_builder.build("PN", args.label, args.g_lr)

    # Move generator to device
    g_model = device_manager.move_to_device(g_model)

    # Build discriminator model if using GAN (only BlueDisc)
    d_model = None
    if use_gan:
        d_builder = DBuilder()
        d_model = d_builder.build("BlueDisc", args.d_lr)

        # Move discriminator to device
        d_model = device_manager.move_to_device(d_model)

    project_root = os.getcwd()

    print("Using local MLflow configuration")
    mlflow_host = "127.0.0.1"
    mlflow_port = 5000

    mlflow.set_tracking_uri(f"http://{mlflow_host}:{mlflow_port}")

    # Set experiment name
    experient_name = f"PN_{args.label}"

    if use_gan:
        experient_name += f"_GAN"
        if data_weight == 0:
            experient_name += "_Data0"
        else:
            experient_name += f"_Data{data_weight}"

    mlflow.set_experiment(experient_name)
    print(f"Experient: {experient_name}")

    # Dynamically load the dataset
    data_class = getattr(sbd, args.dataset)
    data = data_class(sampling_rate=100)

    train, dev, test = data.train_dev_test()
    print(
        f"train: {len(train)}, dev: {len(dev)}, track: {sample_size}, test: {len(test)}"
    )

    train_generator = sbg.GenericGenerator(train)
    dev_generator = sbg.GenericGenerator(dev)

    aug_builder = AugmentationsBuilder(dataset=data)
    augmentations = aug_builder.build()

    train_generator.add_augmentations(augmentations)
    dev_generator.add_augmentations(augmentations)

    g = torch.Generator()
    g.manual_seed(seed_value)

    if args.batch_size is None:
        raise ValueError("Batch size must be specified and should be an integer.")

    num_workers = os.cpu_count() or 1

    train_loader = DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_manager.worker_init_fn,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        generator=g,
    )

    dev_loader = DataLoader(
        dev_generator,
        batch_size=sample_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_manager.worker_init_fn,
        pin_memory=True,
        generator=g,
    )

    # Get sample data from the dev set
    test_samples = next(iter(dev_loader))
    trace_name = test_samples.pop("trace_name", "")
    label = g_model.reorder_label_phase(test_samples)

    with mlflow.start_run():
        current_run = mlflow.active_run()
        run_id = current_run.info.run_id

        # Initialize logger
        logger = MLFlowLogger(
            run_id=run_id,
            mlflow_host=mlflow_host,
            mlflow_port=mlflow_port
        )

        # Initialize GANModel with logger
        gan_model = GANModel(
            gan_type=gan_type,
            generator=g_model,
            discriminator=d_model,
            g_data_weight=data_weight,
            gan_loss_weight=gan_loss_weight,
            logger=logger,
        )
        print(f"Models on device: {device_manager.device}")
        print("compile model")
        gan_model = torch.compile(gan_model)


        # Log parameters
        logger.log_param("gan_type", gan_type)
        logger.log_param("g_model", "PN")
        logger.log_param("d_model", "BlueDisc" if d_model else None)
        logger.log_param("dataset", args.dataset)
        logger.log_param("sample_size", sample_size)
        logger.log_param("batch_size", batch_size)
        logger.log_param("g_lr", g_model.lr)
        logger.log_param("d_lr", d_model.lr if d_model else None)
        logger.log_param("g_data_weight", data_weight)
        logger.log_param("gan_loss_weight", gan_loss_weight)
        logger.log_param("max_steps", max_steps)

        # Log sample data to artifacts (synchronously)
        logger.log_hdf5(
            test_samples["X"].numpy(), data_split="track", data_type="waveform", step=0
        )
        logger.log_hdf5(
            label.numpy(), data_split="track", data_type="label", step=0
        )
        logger.log_text(
            trace_name, data_split="track", data_type="trace_name", step=0
        )

        # Start training
        gan_model.fit(
            train_loader=train_loader,
            test_samples=test_samples,
            max_steps=max_steps,
        )
