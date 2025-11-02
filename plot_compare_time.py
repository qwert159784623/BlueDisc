import argparse
import os
import sys

from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.signal import find_peaks

# Set global font sizes for matplotlib
fontsize = 20
plt.rcParams.update(
    {
        "font.size": fontsize,  # Default font size
        "axes.titlesize": fontsize,  # Title font size
        "axes.labelsize": fontsize,  # Axis label font size
        "xtick.labelsize": fontsize,  # X-axis tick label size
        "ytick.labelsize": fontsize,  # Y-axis tick label size
        "figure.titlesize": fontsize,  # Figure title font size
    }
)


def plot_waveform(sample_id):
    # Plot the waveform and the final stacked image
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 15), gridspec_kw={"height_ratios": [1, 10]}
    )

    sample_waveform = f"{base_path}/track/waveform/waveform_0000000.h5"
    with h5py.File(sample_waveform, "r") as f:
        waveform_data = f["data"][sample_id]

    # Create time axis in seconds (assuming 100Hz sampling rate)
    time_axis = np.arange(len(waveform_data[0])) / sample_rate

    ax1.plot(time_axis, waveform_data[2], color="gray", label="E", linewidth=0.1)
    ax1.plot(time_axis, waveform_data[1], color="gray", label="N", linewidth=0.1)
    ax1.plot(time_axis, waveform_data[0], color="black", label="Z", linewidth=0.5)
    ax1.margins(x=0)

    # Load the data from each file
    all_data = []
    pred_p_picks = []
    pred_s_picks = []
    step_count = 0
    for step in tqdm(range(max_step)):
        pred_path = f"{base_path}/track/prediction/prediction_{step:0>7}.h5"
        if not os.path.exists(pred_path):
            break
        with h5py.File(pred_path, "r") as f:
            pred = f["data"][sample_id]

        p_peaks, _ = find_peaks(pred[0], distance=100, height=confidence)
        pred_p_picks.append(p_peaks)
        s_peaks, _ = find_peaks(pred[1], distance=100, height=confidence)
        pred_s_picks.append(s_peaks)
        all_data.append(pred)
        step_count += 1
        if step_count >= max_step:
            break

    # Convert data to RGB images
    image_size = (1, len(all_data[0][0]), 3)  # 1 pixel height, full width
    horizontal_images = []

    for dataset in all_data:
        rgb_data = np.array(
            dataset[:3]
        ).T  # Take the first three lists as RGB channels and transpose
        brg_data = rgb_data[:, [1, 2, 0]]  # Swap RGB to BRG
        brg_data[:, 1] = 0  # Set G channel to 0
        brg_data = (brg_data - np.min(brg_data)) / (
            np.max(brg_data) - np.min(brg_data)
        )  # Normalize to 0-1
        brg_data = (
            (brg_data * 255).astype(np.uint8).reshape(image_size)
        )  # Scale to 0-255 and reshape
        horizontal_images.append(brg_data)

    # Stack images vertically
    stacked_image = np.vstack(horizontal_images)

    # Plot the final stacked image on the second subplot
    ax2.imshow(stacked_image, aspect="auto")

    label_file = f"{base_path}/track/label/label_0000000.h5"
    with h5py.File(label_file, "r") as f:
        label_data = f["data"][sample_id]

    label_data = np.array(label_data)
    label_p_peaks, _ = find_peaks(label_data[0], distance=100, height=confidence)
    label_s_peaks, _ = find_peaks(label_data[1], distance=100, height=confidence)

    index_threshold = int(time_threshold * sample_rate)

    # Convert peak indices to time in seconds for vertical lines
    label_p_times = label_p_peaks / sample_rate
    label_s_times = label_s_peaks / sample_rate

    ax1.vlines(label_p_times, ymin=-1, ymax=1, color="blue", label="P")
    ax1.vlines(label_s_times, ymin=-1, ymax=1, color="red", label="S")
    ax1.set_title(f"{experiment_name}, {training_dataset} track {sample_id:0>3}")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")

    for label_p in label_p_peaks:
        ax2.axvline(label_p, color="gray")
        for step, peaks in enumerate(pred_p_picks):
            for p in peaks:
                color = "gray"
                if abs(p - label_p) < index_threshold:
                    color = "yellow"
                ax2.scatter(p, step, color=color, s=5)

    for label_s in label_s_peaks:
        ax2.axvline(label_s, color="gray")
        for step, peaks in enumerate(pred_s_picks):
            for s in peaks:
                color = "gray"
                if abs(s - label_s) < index_threshold:
                    color = "yellow"
                ax2.scatter(s, step, color=color, s=5)

    ax2.set_ylabel("Training Step")
    ax2.set_xlabel("Time (seconds)")
    ax2.xaxis.set_visible(True)

    # Set x-axis ticks to show time in seconds
    ax2_xlim = ax2.get_xlim()
    xticks = np.arange(0, ax2_xlim[1], step=int(sample_rate * 5))  # Every 5 seconds
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{int(tick/sample_rate)}" for tick in xticks])

    ax2.tick_params(axis="y", rotation=45)

    # Add legend to bottom subplot in bottom right corner
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Line2D([0], [0], color="gray", lw=1, label="True P/S label"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            linestyle='None',
            label="Peak > 0.7",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="yellow",
            markersize=8,
            linestyle='None',
            label="Precise Peak < 0.1s",
        ),
        Patch(facecolor="blue", label="P Phase"),
        Patch(facecolor="red", label="S Phase"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", framealpha=0.3, labelcolor='lightgray')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)  # Reduce vertical space between subplots
    os.makedirs(f"{base_path}/sample_history", exist_ok=True)
    plt.savefig(
        f"{base_path}/sample_history/{experiment_name}_sample_{sample_id:0>3}.png",
        dpi=300,  # Higher DPI for better quality
    )
    plt.close()
    print(f"Saved artifacts/sample_history/{experiment_name}_sample_{sample_id:0>3}.png")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID for the trained model")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps to plot")
    parser.add_argument("--sample-ids", type=str, default=None, help="Comma-separated list of sample IDs to plot (e.g., '0,1,2'). If not specified, all samples will be plotted.")

    args = parser.parse_args()

    mlflow_host = '0.0.0.0'
    mlflow_port = 5000
    client = mlflow.MlflowClient(f"http://{mlflow_host}:{mlflow_port}")

    run_id = args.run_id
    current_run = client.get_run(run_id)
    experiment_id = current_run.info.experiment_id
    experiment_name = client.get_experiment(experiment_id).name
    training_dataset = current_run.data.params["dataset"]

    base_path = f"mlruns/{experiment_id}/{run_id}/artifacts"

    sample_rate = 100
    time_threshold = 0.1  # in seconds
    confidence = 0.7

    # Auto-detect max_steps from prediction files if not specified
    if args.max_steps is None:
        import glob
        prediction_path = os.path.join(base_path, "track", "prediction")
        if os.path.exists(prediction_path):
            prediction_files = glob.glob(os.path.join(prediction_path, "prediction_*.h5"))
            if prediction_files:
                max_step = len(prediction_files)
                print(f"Auto-detected {max_step} prediction files")
            else:
                max_step = 10000
                print(f"No prediction files found. Using default max_step={max_step}")
        else:
            max_step = 10000
            print(f"Prediction path not found. Using default max_step={max_step}")
    else:
        max_step = args.max_steps

    # Auto-detect sample_size from trace_name file if sample_ids not specified
    if args.sample_ids:
        sample_ids = [int(sample_id) for sample_id in args.sample_ids.split(",")]
    else:
        trace_name_file = os.path.join(base_path, "track", "trace_name", "trace_name_0000000.txt")
        if os.path.exists(trace_name_file):
            with open(trace_name_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            sample_size = len(lines)
            print(f"Auto-detected {sample_size} samples from trace_name file")
        else:
            sample_size = 100
            print(f"Trace name file not found. Using default sample_size={sample_size}")
        sample_ids = range(sample_size)

    for ids in sample_ids:
        plot_waveform(ids)