import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlflow
import numpy as np
from scipy.signal import find_peaks


def _color_palette(color=1, shade=1):
    """
    Return a color palette form a selected color and shade level.

    :param int color: (Optional.) 0=Blue, 1=Deep Orange, 2=Green, 3=Purple, default is 1.
    :param int shade: (Optional.) 0=light, 1=light medium, 2=regular, 3=dark, default is 1.
    :rtype: str
    :return: Hex color code.
    """
    palette = [
        ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"],  # Blue
        ["#FFAB91", "#FF7043", "#E64A19", "#BF360C"],  # Deep Orange
        ["#A5D6A7", "#66BB6A", "#388E3C", "#1B5E20"],  # Green
        ["#E1BEE7", "#AB47BC", "#7B1FA2", "#4A148C"],  # Purple
    ]

    return palette[color][shade]


def plot_sample_for_steps(sample_id):
    """Plot all specified steps for a given sample_id in one figure with shared x-axis"""
    current_run = client.get_run(args.run_id)
    experiment_id = current_run.info.experiment_id
    experiment_name = client.get_experiment(experiment_id).name
    training_dataset = current_run.data.params["dataset"]

    data_split = args.data_split
    output_folder = "sample_plot"
    batch = args.batch

    base_path = f"mlruns/{experiment_id}/{args.run_id}/artifacts"
    output_path = os.path.join(base_path, args.data_split, output_folder)
    os.makedirs(output_path, exist_ok=True)

    sample_rate = 100
    confidence = 0.7

    # Load waveform and label data
    sample_waveform = os.path.join(
        base_path, args.data_split, "waveform", "waveform_0000000.h5"
    )
    with h5py.File(sample_waveform, "r") as f:
        waveform_data = f["data"][sample_id]

    label_file = os.path.join(base_path, args.data_split, "label", "label_0000000.h5")
    with h5py.File(label_file, "r") as f:
        run_label_data = f["data"][sample_id]

    # Calculate number of subplots: 1 waveform plot + prediction plots for each step
    num_steps = len(args.steps)
    total_subplots = 1 + num_steps

    dpi = 150
    width_inches = 10
    height_inches = (3 + (num_steps * 2.5)) / 2

    # Create figure and GridSpec
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)

    height_ratios = [1] + [1] * num_steps
    hspace_values = [0.3] + [0.05] * (num_steps - 1)

    gs = gridspec.GridSpec(
        total_subplots, 1, height_ratios=height_ratios, hspace=0.1
    )

    # Waveform plot at the top
    ax_waveform = fig.add_subplot(gs[0])

    # Prediction plots share x-axis
    pred_axes = []
    for i in range(num_steps):
        if i == 0:
            ax = fig.add_subplot(gs[i + 1])
        else:
            ax = fig.add_subplot(gs[i + 1], sharex=pred_axes[0])
        pred_axes.append(ax)

    # Create time axis in seconds
    time_axis = np.arange(len(waveform_data[0])) / sample_rate

    # First subplot: waveform plot
    ax_waveform.plot(
        time_axis, waveform_data[2], color="gray", label="E", linewidth=0.1
    )
    ax_waveform.plot(
        time_axis, waveform_data[1], color="gray", label="N", linewidth=0.2
    )
    ax_waveform.plot(
        time_axis, waveform_data[0], color="black", label="Z", linewidth=0.5
    )
    ax_waveform.margins(x=0)
    ax_waveform.set_ylabel("Amplitude", fontsize=10)
    ax_waveform.set_title(
        f"{experiment_name}, {training_dataset}, {data_split} {sample_id:0>3}",
        fontsize=14,
    )
    ax_waveform.tick_params(axis="both", which="major", labelsize=12)

    # Find label peaks
    label_p_peaks, _ = find_peaks(run_label_data[0], distance=100, height=confidence)
    label_s_peaks, _ = find_peaks(run_label_data[1], distance=100, height=confidence)

    # Convert peak positions to seconds
    label_p_peaks_time = label_p_peaks / sample_rate
    label_s_peaks_time = label_s_peaks / sample_rate

    # Add vertical lines for label peaks in waveform plot
    ax_waveform.vlines(
        label_p_peaks_time,
        ymin=-1,
        ymax=1,
        color=_color_palette(0, 0),
        alpha=0.7,
        label=f"P Label",
    )
    ax_waveform.vlines(
        label_s_peaks_time,
        ymin=-1,
        ymax=1,
        color=_color_palette(1, 0),
        alpha=0.7,
        label=f"S Label",
    )

    ax_waveform.legend(loc="upper right", fontsize=10, framealpha=0.7, ncol=2)

    if len(args.steps) == 1:
        ax_waveform.spines["bottom"].set_visible(False)
        ax_waveform.tick_params(axis="x", bottom=False, labelbottom=False)

    # Create prediction plots for each step
    for i, step in enumerate(args.steps):
        ax_pred = pred_axes[i]

        # Load prediction data
        if data_split == "track":
            pred_file = f"{base_path}/{data_split}/prediction/prediction_{step:0>7}.h5"
        else:
            pred_file = f"{base_path}/{data_split}/prediction/prediction_{batch:0>7}.h5"

        with h5py.File(pred_file, "r") as f:
            pred_data = f["data"][sample_id]

        pred_data = np.array(pred_data)

        # Plot labels on each prediction subplot
        ax_pred.plot(
            time_axis,
            run_label_data[0],
            color=_color_palette(0, 0),
            label="P Label",
            linewidth=1,
            alpha=0.7,
        )
        ax_pred.plot(
            time_axis,
            run_label_data[1],
            color=_color_palette(1, 0),
            label="S Label",
            linewidth=1,
            alpha=0.7,
        )
        ax_pred.plot(
            time_axis,
            run_label_data[2],
            color=_color_palette(2, 0),
            label="N/D Label",
            linewidth=1,
            alpha=0.7,
        )

        # Plot predictions
        ax_pred.plot(
            time_axis,
            pred_data[0],
            color=_color_palette(0, 1),
            label="P Pred",
            linewidth=1,
        )
        ax_pred.plot(
            time_axis,
            pred_data[1],
            color=_color_palette(1, 1),
            label="S Pred",
            linewidth=1,
        )
        ax_pred.plot(
            time_axis,
            pred_data[2],
            color=_color_palette(2, 1),
            label="N/D Pred",
            linewidth=1,
        )

        ax_pred.margins(x=0)
        ax_pred.set_ylabel(f"Step {step}", fontsize=10)
        ax_pred.tick_params(axis="both", which="major", labelsize=10)

        # Hide x-axis labels and ticks for all but the last subplot
        if i < len(args.steps) - 1:
            ax_pred.tick_params(axis="x", labelbottom=False)
            ax_pred.spines["bottom"].set_visible(False)
            ax_pred.tick_params(axis="x", bottom=False)
        else:
            ax_pred.set_xlabel("Time (seconds)", fontsize=14)

        # Remove borders for seamless appearance
        if len(args.steps) == 1:
            ax_pred.spines["top"].set_visible(False)
        elif i > 0:
            ax_pred.spines["top"].set_visible(False)
        if i < len(args.steps) - 1:
            ax_pred.spines["bottom"].set_visible(False)

        # Show legend only on the last prediction plot
        if i == len(args.steps) - 1:
            ax_pred.legend(loc="center right", fontsize=9, framealpha=0.7, ncol=2)

    # Set x-axis range if time range is specified
    if args.time_range is not None:
        start_time, end_time = args.time_range
        ax_waveform.set_xlim(start_time, end_time)
        for ax_pred in pred_axes:
            ax_pred.set_xlim(start_time, end_time)

    plt.tight_layout()

    # Adjust spacing between waveform and prediction plots
    gs.update(hspace=0.08)

    pos_waveform = ax_waveform.get_position()
    pos_first_pred = pred_axes[0].get_position()

    # Fine-tune prediction plot positions
    for ax_pred in pred_axes:
        pos = ax_pred.get_position()
        extra_offset = 0.02 if len(args.steps) == 1 else 0
        new_pos = [
            pos.x0,
            pos.y0 - 0.04 - extra_offset,
            pos.width,
            pos.height,
        ]
        ax_pred.set_position(new_pos)

    if data_split == "track":
        save_path = os.path.join(base_path, data_split, output_folder)
    else:
        save_path = os.path.join(base_path, data_split, output_folder)

    os.makedirs(save_path, exist_ok=True)

    plt.savefig(
        f"{save_path}/sample_{sample_id:0>3}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()

    print(f"artifaces/{data_split}/sample_plot/sample_{sample_id:0>3}.png with steps {args.steps} plotted.")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID for the trained model")
    parser.add_argument("--sample-ids", type=int, nargs="+", default=None, help="List of sample IDs to plot. If not specified, all samples will be plotted.")
    parser.add_argument(
        "--batch", type=int, default=0, help="Batch number for test data"
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", required=True, help="Specific steps to plot"
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="track",
        choices=["track", "train", "dev", "test"],
        help="Data split to plot",
    )
    parser.add_argument(
        "--time-range",
        type=float,
        nargs=2,
        default=None,
        help="Time range to zoom in [start_time, end_time] in seconds"
    )

    args = parser.parse_args()

    mlflow_host = '0.0.0.0'
    mlflow_port = 5000
    client = mlflow.MlflowClient(f"http://{mlflow_host}:{mlflow_port}")

    # Get run info to locate trace_name file
    current_run = client.get_run(args.run_id)
    experiment_id = current_run.info.experiment_id

    # Auto-detect max_sample from trace_name file if sample_ids not specified
    if args.sample_ids is None or len(args.sample_ids) == 0:
        trace_name_file = f"mlruns/{experiment_id}/{args.run_id}/artifacts/{args.data_split}/trace_name/trace_name_0000000.txt"

        if os.path.exists(trace_name_file):
            with open(trace_name_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            max_sample = len(lines)
            print(f"Auto-detected {max_sample} samples from trace_name file")
        else:
            # Fallback to default values
            if args.data_split == "track":
                max_sample = 100
            else:
                max_sample = 1000
            print(f"Trace name file not found. Using default max_sample={max_sample}")

        args.sample_ids = range(0, max_sample)

    for ids in args.sample_ids:
        plot_sample_for_steps(ids)
