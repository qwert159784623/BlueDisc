import argparse
import os
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
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


def _get_time_array():
    time_array = np.arange(3001)
    time_array = time_array * 0.01
    return time_array


def plot_sample(
    step,
):
    # Collect all run IDs
    run_ids = [args.run_id, args.run_id2, args.run_id3]
    run_ids = [rid for rid in run_ids if rid is not None]  # Filter out None values

    # Set DPI and figsize to ensure output pixels are even numbers
    dpi = 150
    width_inches = 10  # 12 * 100 = 1200 pixels (even number)

    # Create subplots: 1 waveform plot + n label plots
    num_runs = len(run_ids)
    height_inches = 3 + 2.5 * num_runs
    # Ensure height is also even pixels
    height_inches = round(height_inches * 2) / 2  # Round to nearest 0.5

    fig, axes = plt.subplots(
        num_runs + 1, 1, figsize=(width_inches, height_inches), dpi=dpi
    )

    # If there's only one subplot, ensure axes is a list
    if num_runs == 0:
        return
    if num_runs == 1:
        axes = [axes[0], axes[1]]

    # Create time axis (in seconds)
    time_axis = np.arange(len(waveform_data[0])) / sample_rate

    # First subplot: waveform plot (same for all runs)
    ax_waveform = axes[0]
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
    ax_waveform.set_xlabel("Time (s)", fontsize=14)
    ax_waveform.set_ylabel("Amplitude", fontsize=14)
    ax_waveform.set_title(
        f"{training_dataset}, {data_split} {sample_id:0>3}, Confidence: {confidence}, Step: {step:0>7} ",
        fontsize=16,
    )
    ax_waveform.tick_params(axis="both", which="major", labelsize=12)

    # Find label peaks
    label_p_peaks, _ = find_peaks(run_label_data[0], distance=100, height=confidence)
    label_s_peaks, _ = find_peaks(run_label_data[1], distance=100, height=confidence)

    # Convert peak positions to seconds
    label_p_peaks_time = label_p_peaks / sample_rate
    label_s_peaks_time = label_s_peaks / sample_rate

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
    ax_waveform.legend(bbox_to_anchor=(1.01, 1.03), loc="upper left", fontsize=12)

    # Create label plots for each run
    for i, run_id in enumerate(run_ids):
        ax_label = axes[i + 1]

        # Get current run information
        current_run = client.get_run(run_id)
        experiment_id = current_run.info.experiment_id
        experiment_name = client.get_experiment(experiment_id).name

        # Get current run path
        run_base_path = f"mlruns/{experiment_id}/{run_id}/artifacts"

        # Plot labels
        ax_label.plot(
            time_axis,
            run_label_data[0],
            color=_color_palette(0, 0),
            label="P Label",
            linewidth=1,
        )
        ax_label.plot(
            time_axis,
            run_label_data[1],
            color=_color_palette(1, 0),
            label="S Label",
            linewidth=1,
        )
        ax_label.plot(
            time_axis,
            run_label_data[2],
            color=_color_palette(2, 0),
            label="N/D Label",
            linewidth=1,
        )
        ax_label.margins(x=0)
        ax_label.set_xlabel("Time (s)", fontsize=14)
        ax_label.set_ylabel("Probability", fontsize=14)
        ax_label.tick_params(axis="both", which="major", labelsize=12)

        # Load prediction data
        if data_split == "track":
            pred_file = (
                f"{run_base_path}/{data_split}/prediction/prediction_{step:0>7}.h5"
            )
        else:
            pred_file = (
                f"{run_base_path}/{data_split}/prediction/prediction_{batch:0>7}.h5"
            )

        with h5py.File(pred_file, "r") as f:
            pred_data = f["data"][sample_id]

        pred_data = np.array(pred_data)

        # Plot predictions
        ax_label.plot(
            time_axis,
            pred_data[0],
            color=_color_palette(0, 1),
            label="P Pred",
            linewidth=1,
        )
        ax_label.plot(
            time_axis,
            pred_data[1],
            color=_color_palette(1, 1),
            label="S Pred",
            linewidth=1,
        )
        ax_label.plot(
            time_axis,
            pred_data[2],
            color=_color_palette(2, 1),
            label="N/D Pred",
            linewidth=1,
        )

        # Find prediction peaks
        pred_p_peaks, _ = find_peaks(pred_data[0], distance=100, height=confidence)
        pred_s_peaks, _ = find_peaks(pred_data[1], distance=100, height=confidence)

        # Convert prediction peak positions to seconds
        pred_p_peaks_time = pred_p_peaks / sample_rate
        pred_s_peaks_time = pred_s_peaks / sample_rate

        # Y-axis positions for picks adjusted by number of runs
        row_offset = {
            1: {  # 1 run: middle row
                "ymax": [0.5],
                "ymin": [-0.5],
            },
            2: {  # 2 runs: top, bottom rows
                "ymax": [0.8, 0.3],
                "ymin": [-0.3, -0.8],
            },
            3: {  # 3 runs: top, middle, bottom rows
                "ymax": [0.8, 0.5, 0.3],
                "ymin": [-0.3, -0.5, -0.8],
            },
        }

        ax_waveform.vlines(
            pred_p_peaks_time,
            ymin=row_offset[num_runs]["ymin"][i],
            ymax=row_offset[num_runs]["ymax"][i],
            color=_color_palette(0, i + 1),
            alpha=0.7,
        )
        ax_waveform.vlines(
            pred_s_peaks_time,
            ymin=row_offset[num_runs]["ymin"][i],
            ymax=row_offset[num_runs]["ymax"][i],
            color=_color_palette(1, i + 1),
            alpha=0.7,
        )

        # Set label plot title
        ax_label.set_title(f"{experiment_name}", fontsize=15)

        # Add legend
        ax_label.legend(bbox_to_anchor=(1.01, 1.03), loc="upper left", fontsize=12)

    plt.tight_layout()

    if data_split == "track":
        save_path = os.path.join(
            base_path, data_split, output_folder, f"step_{step:0>7}"
        )
    else:
        save_path = os.path.join(
            base_path, data_split, output_folder, f"batch_{batch:0>7}"
        )

    os.makedirs(save_path, exist_ok=True)

    plt.savefig(
        f"{save_path}/compare_sample_{sample_id:0>3}_step_{step:0>7}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()

    print(f" artifacts/{data_split}/{output_folder}/compare_sample_{sample_id:0>3}_step_{step:0>7}.png is plotted.")

    sys.stdout.flush()


def animate_sample():
    # Set folder path and output file name
    input_pattern = f"{base_path}/{data_split}/{output_folder}/step_*/compare_sample_{sample_id:0>3}_step_*.png"
    output_file = f"{base_path}/{data_split}/{output_folder}/compare_{experiment_name}_sample_{sample_id:0>3}.mp4"

    fps = 60  # Frame rate

    # Build ffmpeg command
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-framerate",
        str(fps),  # Set frame rate
        "-pattern_type",
        "glob",  # Use glob pattern to match files
        "-i",
        input_pattern,  # Image input pattern
        "-c:v",
        "libx264",  # Use H.264 encoding
        "-crf",
        "18",  # Video quality
        "-pix_fmt",
        "yuv420p",  # Ensure compatibility
        output_file,  # Output file
    ]

    # Execute ffmpeg command
    try:
        subprocess.run(command, check=True)
        print(f"Comparison video successfully output to: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Execution failed:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True, help="MLflow run ID for the trained model")
    parser.add_argument("--run-id2", type=str, help="Second MLflow run ID for comparison")
    parser.add_argument("--run-id3", type=str, help="Third MLflow run ID for comparison")
    parser.add_argument("--sample-ids", type=int, nargs="+", help="List of sample IDs to plot")
    parser.add_argument(
        "--batch", type=int, default=0, help="Batch number for test data"
    )
    parser.add_argument("--min-steps", type=int, default=0, help="Minimum step to start plotting from")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum step to plot up to")
    parser.add_argument(
        "--data-split",
        type=str,
        default="track",
        choices=["track", "train", "dev", "test"],
        help="Data split to plot",
    )
    parser.add_argument("--animation", action="store_true", help="Generate animation")

    args = parser.parse_args()

    mlflow_host = '0.0.0.0'
    mlflow_port = 5000
    client = mlflow.MlflowClient(f"http://{mlflow_host}:{mlflow_port}")

    # Get run info once at the beginning
    first_run_id = args.run_id
    current_run = client.get_run(first_run_id)
    experiment_id = current_run.info.experiment_id
    experiment_name = client.get_experiment(experiment_id).name
    training_dataset = current_run.data.params["dataset"]

    # Auto-detect max_sample from trace_name file if sample_ids not specified
    if args.sample_ids is None or len(args.sample_ids) == 0:
        trace_name_file = f"mlruns/{experiment_id}/{first_run_id}/artifacts/{args.data_split}/trace_name/trace_name_0000000.txt"

        if os.path.exists(trace_name_file):
            with open(trace_name_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]  # Count non-empty lines
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

    for sample_id in args.sample_ids:

        data_split = args.data_split
        output_folder = "compare_plot"
        batch = args.batch

        base_path = f"mlruns/{experiment_id}/{first_run_id}/artifacts"
        output_path = os.path.join(base_path, args.data_split, output_folder)
        os.makedirs(output_path, exist_ok=True)

        # Auto-detect max_steps if not specified
        if args.max_steps is None:
            import glob
            import re
            prediction_path = os.path.join(base_path, data_split, "prediction")
            if os.path.exists(prediction_path):
                prediction_files = glob.glob(os.path.join(prediction_path, "prediction_*.h5"))
                if prediction_files:
                    # Extract step numbers from filenames
                    step_numbers = []
                    for f in prediction_files:
                        match = re.search(r'prediction_(\d+)\.h5', os.path.basename(f))
                        if match:
                            step_numbers.append(int(match.group(1)))

                    if step_numbers:
                        # Files are numbered from 0 consecutively
                        # max_steps is exclusive upper bound for range()
                        # So if we have 100 files (0-99), max_steps = 100
                        max_steps = len(prediction_files)
                        print(f"Auto-detected {len(prediction_files)} prediction files.")
                        print(f"Step numbers: {min(step_numbers)} to {max(step_numbers)}")
                        print(f"Using range({args.min_steps}, {max_steps})")
                    else:
                        max_steps = 1000
                        print(f"Could not parse step numbers from files. Using default max_steps={max_steps}")
                else:
                    max_steps = 1000
                    print(f"No prediction files found. Using default max_steps={max_steps}")
            else:
                max_steps = 1000
                print(f"Prediction path not found: {prediction_path}. Using default max_steps={max_steps}")
        else:
            max_steps = args.max_steps

        sample_rate = 100
        time_threshold = 0.1  # in seconds
        confidence = 0.7

        # Use run's waveform and label (same for all runs)
        sample_waveform = os.path.join(
            base_path, args.data_split, "waveform", "waveform_0000000.h5"
        )
        with h5py.File(sample_waveform, "r") as f:
            waveform_data = f["data"][sample_id]

        label_file = os.path.join(
            base_path, args.data_split, "label", "label_0000000.h5"
        )
        with h5py.File(label_file, "r") as f:
            run_label_data = f["data"][sample_id]

        for step in range(args.min_steps, max_steps):
            plot_sample(step)

        if args.animation:
            animate_sample()
