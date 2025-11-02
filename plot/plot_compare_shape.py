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
    # 使用單一 run 來設置基本參數和路徑
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

    # 載入 waveform 和 label 數據
    sample_waveform = os.path.join(
        base_path, args.data_split, "waveform", "waveform_0000000.h5"
    )
    with h5py.File(sample_waveform, "r") as f:
        waveform_data = f["data"][sample_id]

    label_file = os.path.join(base_path, args.data_split, "label", "label_0000000.h5")
    with h5py.File(label_file, "r") as f:
        run_label_data = f["data"][sample_id]

    # 計算子圖數量：1個波形圖 + steps數量的預測圖
    num_steps = len(args.steps)
    total_subplots = 1 + num_steps

    # 設定 DPI 和 figsize
    dpi = 150
    width_inches = 10
    height_inches = (3 + (num_steps * 2.5)) / 2  # 將整體高度減少一半

    # 創建 figure 和 GridSpec
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)

    # 使用 GridSpec 來控制布局
    # height_ratios: 地震波圖片佔 1 個單位（縮減一半），每個預測圖佔 1 個單位
    height_ratios = [1] + [1] * num_steps
    # 設定間距：地震波圖片下方有較大間距，預測圖之間間距較小
    hspace_values = [0.3] + [0.05] * (
        num_steps - 1
    )  # 地震波與預測圖間距 0.3，預測圖間距 0.05

    gs = gridspec.GridSpec(
        total_subplots, 1, height_ratios=height_ratios, hspace=0.1
    )  # 整體間距縮減

    # 波形圖佔據頂部
    ax_waveform = fig.add_subplot(gs[0])

    # 預測圖共用 x 軸
    pred_axes = []
    for i in range(num_steps):
        if i == 0:
            ax = fig.add_subplot(gs[i + 1])
        else:
            ax = fig.add_subplot(gs[i + 1], sharex=pred_axes[0])
        pred_axes.append(ax)

    # 創建時間軸（以秒為單位）
    time_axis = np.arange(len(waveform_data[0])) / sample_rate

    # 第一個子圖：波形圖（共用）
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

    # 找到标签峰值
    label_p_peaks, _ = find_peaks(run_label_data[0], distance=100, height=confidence)
    label_s_peaks, _ = find_peaks(run_label_data[1], distance=100, height=confidence)

    # 將峰值位置轉換為秒
    label_p_peaks_time = label_p_peaks / sample_rate
    label_s_peaks_time = label_s_peaks / sample_rate

    # 在波形图中添加 label peaks 的垂直线标记
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

    # 如果只有一個step，去掉波形圖的下邊框和x軸標籤
    if len(args.steps) == 1:
        ax_waveform.spines["bottom"].set_visible(False)
        ax_waveform.tick_params(axis="x", bottom=False, labelbottom=False)

    # 為每個 step 創建預測圖
    for i, step in enumerate(args.steps):
        ax_pred = pred_axes[i]

        # 加載預測數據
        if data_split == "track":
            pred_file = f"{base_path}/{data_split}/prediction/prediction_{step:0>7}.h5"
        else:
            pred_file = f"{base_path}/{data_split}/prediction/prediction_{batch:0>7}.h5"

        with h5py.File(pred_file, "r") as f:
            pred_data = f["data"][sample_id]

        pred_data = np.array(pred_data)

        # 繪製標籤（在每個預測圖上）
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

        # 繪製預測
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
        # 只顯示 step 信息在 y 軸標題
        ax_pred.set_ylabel(f"Step {step}", fontsize=10)
        ax_pred.tick_params(axis="both", which="major", labelsize=10)

        # 隱藏除了最後一個子圖外的 x 軸標籤和刻度
        if i < len(args.steps) - 1:
            ax_pred.tick_params(axis="x", labelbottom=False)
            # 移除頂部邊框以創造無縫效果
            ax_pred.spines["bottom"].set_visible(False)
            ax_pred.tick_params(axis="x", bottom=False)
        else:
            ax_pred.set_xlabel("Time (seconds)", fontsize=14)

        # 移除除了第一個和最後一個子圖外的上下邊框
        # 如果只有一個step，則移除上邊框（像最後一張一樣）
        if len(args.steps) == 1:
            ax_pred.spines["top"].set_visible(False)
        elif i > 0:
            ax_pred.spines["top"].set_visible(False)
        if i < len(args.steps) - 1:
            ax_pred.spines["bottom"].set_visible(False)

        # 只在最後一個預測圖顯示圖例
        if i == len(args.steps) - 1:
            ax_pred.legend(loc="center right", fontsize=9, framealpha=0.7, ncol=2)

    # 如果指定了時間範圍，設置所有子圖的 x 軸範圍
    if args.time_range is not None:
        start_time, end_time = args.time_range
        ax_waveform.set_xlim(start_time, end_time)
        for ax_pred in pred_axes:
            ax_pred.set_xlim(start_time, end_time)

    # 使用 tight_layout 來優化整體布局，保持預測子圖寬度一致
    plt.tight_layout()

    # 手動調整地震波圖片與第一個預測圖之間的間距
    gs.update(hspace=0.08)  # 減少整體間距

    # 單獨增加波形圖與預測圖之間的間距（縮減一半）
    pos_waveform = ax_waveform.get_position()
    pos_first_pred = pred_axes[0].get_position()

    # 向下移動預測圖區域（縮減間距）
    for ax_pred in pred_axes:
        pos = ax_pred.get_position()
        # 如果只有一個step，額外向下移動更多一點
        extra_offset = 0.02 if len(args.steps) == 1 else 0
        new_pos = [
            pos.x0,
            pos.y0 - 0.04 - extra_offset,
            pos.width,
            pos.height,
        ]  # 向下移動 0.04（原來的一半）+ 單張圖額外偏移
        ax_pred.set_position(new_pos)

    # 儲存圖片 - 簡化檔名，不包含 steps
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

    print(f"Sample {sample_id:0>3} with steps {args.steps} plotted.")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+")
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

    if args.data_split == "track":
        max_sample = 100
    else:
        max_sample = 1000

    if args.sample_ids is None or len(args.sample_ids) == 0:
        args.sample_ids = range(0, max_sample)

    for ids in args.sample_ids:
        plot_sample_for_steps(ids)
