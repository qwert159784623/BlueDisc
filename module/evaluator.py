import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist


def get_picks(batch_data, confidence=0.7, distance=100):
    batch_picks = []
    for batch in batch_data:
        p_peaks, p_properties = find_peaks(
            batch[0], distance=distance, height=confidence
        )
        s_peaks, s_properties = find_peaks(
            batch[1], distance=distance, height=confidence
        )
        batch_picks.append(
            {
                "P": {"peaks": p_peaks, "heights": p_properties["peak_heights"]},
                "S": {"peaks": s_peaks, "heights": s_properties["peak_heights"]},
            }
        )

    return batch_picks


def match_peaks_and_calculate_errors(
    pred_picks,
    label_picks,
    trace_names,
    tolerance=500,
    precision_tolerance=10,
    precision_confidence=0.7,
):
    """
    Match predicted peaks with labeled peaks and calculate errors.

    Args:
        pred_picks: List of predicted peaks
        label_picks: List of labeled peaks
        tolerance: Maximum allowed distance for matching
        precision_tolerance: Distance threshold for a "precise" match (default 10,
            roughly 0.1s at a 100 Hz sampling rate)
        precision_confidence: Minimum peak height required for a precise match

    Returns:
        matched_results: A pandas DataFrame containing matching results and errors
    """
    batch_results = []
    for batch_idx, (pred_batch, label_batch, trace_name) in enumerate(
        zip(pred_picks, label_picks, trace_names)
    ):
        for phase in ["P", "S"]:
            pred_peaks = pred_batch[phase]["peaks"]
            label_peaks = label_batch[phase]["peaks"]
            pred_heights = pred_batch[phase]["heights"]
            label_heights = label_batch[phase]["heights"]

            # If either set of peaks is empty, handle the boundary case
            if len(pred_peaks) == 0 or len(label_peaks) == 0:
                phase_result = {
                    "trace_name": trace_name,
                    "phase": phase,
                    "batch_idx": batch_idx,
                    "matched_pairs": [],
                    "unmatched_pred": list(range(len(pred_peaks))),
                    "unmatched_label": list(range(len(label_peaks))),
                    "position_errors": [],
                    "heights": [],
                    "precise_matches": [],
                    "total_distance": 0,
                    "num_matches": 0,
                    "num_precise_matches": 0,
                    "num_pred_peaks": len(pred_peaks),
                    "num_label_peaks": len(label_peaks),
                }
                batch_results.append(phase_result)
                continue

            # Compute the distance matrix between all predicted and labeled peaks
            pred_positions = pred_peaks.reshape(-1, 1)
            label_positions = label_peaks.reshape(-1, 1)
            distance_matrix = cdist(pred_positions, label_positions, metric="euclidean")

            # Use a greedy algorithm to match peaks (smallest distance first)
            matched_pairs = []
            used_pred = set()
            used_label = set()
            position_errors = []
            heights = []
            precise_matches = []

            # Create a list of (distance, pred_index, label_index) and sort it
            distance_pairs = []
            for i in range(len(pred_peaks)):
                for j in range(len(label_peaks)):
                    distance_pairs.append((distance_matrix[i, j], i, j))

            distance_pairs.sort(key=lambda x: x[0])

            # Greedy matching
            for distance, pred_idx, label_idx in distance_pairs:
                if (
                    distance <= tolerance
                    and pred_idx not in used_pred
                    and label_idx not in used_label
                ):
                    matched_pairs.append((pred_idx, label_idx))
                    used_pred.add(pred_idx)
                    used_label.add(label_idx)

                    # Calculate position error and record peak height
                    pos_error = int(pred_peaks[pred_idx] - label_peaks[label_idx])
                    height = float(pred_heights[pred_idx])

                    position_errors.append(pos_error)
                    heights.append(height)

                    # Determine whether this is a precise match
                    is_precise = (
                        distance <= precision_tolerance
                        and height >= precision_confidence
                    )
                    precise_matches.append(is_precise)

            # Find unmatched peaks
            unmatched_pred = [i for i in range(len(pred_peaks)) if i not in used_pred]
            unmatched_label = [
                i for i in range(len(label_peaks)) if i not in used_label
            ]

            total_distance = sum(
                distance_matrix[pred_idx, label_idx]
                for pred_idx, label_idx in matched_pairs
            )

            phase_result = {
                "trace_name": trace_name,
                "phase": phase,
                "batch_idx": batch_idx,
                "matched_pairs": matched_pairs,
                "unmatched_pred": unmatched_pred,
                "unmatched_label": unmatched_label,
                "position_errors": position_errors,
                "heights": heights,
                "precise_matches": precise_matches,
                "total_distance": total_distance,
                "num_matches": len(matched_pairs),
                "num_precise_matches": sum(precise_matches),
                "num_pred_peaks": len(pred_peaks),
                "num_label_peaks": len(label_peaks),
            }

            batch_results.append(phase_result)

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(batch_results)

    return df