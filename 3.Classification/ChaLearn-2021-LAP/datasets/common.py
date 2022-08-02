import csv
import glob
import os
import pandas as pd

def collect_samples(has_labels, root_path, job_path, sequence_length, temporal_stride, job, label_file_path,
                    retrain_all=False):
    if not has_labels:  # Unlabeled set (validation in stage 1, test in stage 2)
        samples = []
        for video_file in sorted(glob.glob(os.path.join(root_path, job_path, '*_color.mp4'))):
            nframes_file = video_file.replace('color.mp4', 'nframes')
            with open(nframes_file, encoding="utf8") as nff:
                num_frames = int(nff.readline())
            # Center crop frames
            frame_start = (num_frames - sequence_length) // (2 * temporal_stride)
            frame_end = frame_start + sequence_length * temporal_stride
            if frame_start < 0:
                frame_start = 0
            if frame_end > num_frames:
                frame_end = num_frames
            frame_indices = list(range(frame_start, frame_end, temporal_stride))
            while len(frame_indices) < sequence_length:
                # Pad
                frame_indices.append(frame_indices[-1])
            samples.append({
                'path': video_file,
                'label': None,
                'frames': frame_indices
            })
        return samples
    else:
        with open(label_file_path, encoding="utf8") as label_file:
            reader = pd.read_csv(label_file, encoding='utf-8', header=None)
            samples = []
            for name, label, job_type in zip(reader[0], reader[1], reader[2]):
                if job_type == job or (
                        retrain_all and job == 'train'):  # If re-training, we want all samples, not just training
                    video_file = os.path.join(root_path, job_path, name + '_color.mp4')
                    nframes_file = video_file.replace('color.mp4', 'nframes')
                    with open(nframes_file, encoding="utf8") as nff:
                        num_frames = int(nff.readline())
                    # Center crop frames
                    frame_start = (num_frames - sequence_length) // (2 * temporal_stride)
                    frame_end = frame_start + sequence_length * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > num_frames:
                        frame_end = num_frames
                    frame_indices = list(range(frame_start, frame_end, temporal_stride))
                    while len(frame_indices) < sequence_length:
                        # Pad
                        frame_indices.append(frame_indices[-1])
                    samples.append({
                        'path': video_file,
                        'label': label,
                        'frames': frame_indices
                    })
            return samples
