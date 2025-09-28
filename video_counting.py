# video_counting.py
# Helpers for processing videos with tracking (so objects have IDs) and counting when crossing a line.


import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import tempfile
import os


def process_video(video_path, model: YOLO, classes=None, conf_thres=0.25, line_position=0.75):
    """Process a video file and return (annotated_video_path, counts_dict).

    - video_path: path to input video
    - model: loaded YOLO model
    - classes: list of class indices to keep (None = all)
    - conf_thres: detection confidence threshold
    - line_position: relative vertical position for counting (0..1 measured from top)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0


    # output file
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


    line_y = int(height * line_position)


    class_counts = defaultdict(int)
    crossed_ids = set()

    # Keep track of the previous centroid y for each track id so we can detect
    # a crossing event from above -> below (or below -> above if desired).
    last_centroids = dict()


    while True:
        ret, frame = cap.read()
        if not ret:
            break


        # Run tracking (keeps track IDs)
        results = model.track(frame, persist=True, conf=conf_thres, classes=classes)

        # default empty lists so code below doesn't crash when there are no detections
        boxes = []
        track_ids = []
        class_indices = []
        confidences = []
        cls_name_map = {}  # map track_id -> class name for easy lookup

        # results from ultralytics may be empty or have no boxes; guard against that
        if results and len(results) > 0 and getattr(results[0].boxes, "data", None) is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            # draw counting line (draw inside the detection block so it's always visible too)
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)


            # iterate detections and draw them; also prepare mapping for counting
            for (x1_f, y1_f, x2_f, y2_f), track_id, cls_idx, conf in zip(
                boxes, track_ids, class_indices, confidences
            ):
                # make coordinates integers (important for cv2 drawing)
                x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # cls_name resolved from model.names
                cls_name = model.names[int(cls_idx)]
                cls_name_map[int(track_id)] = cls_name

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id} {cls_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # counting logic per object: compare current centroid with last centroid
                prev_cy = last_centroids.get(track_id, None)

                # If previously above (or unknown) and now below -> count
                if prev_cy is not None:
                    if prev_cy <= line_y and cy > line_y and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        class_counts[cls_name] += 1
                else:
                    # If we have no previous position, don't immediately count unless
                    # you want to count objects that first appear already below the line.
                    # (This branch intentionally does nothing.)
                    pass

                # update last centroid position for this id
                last_centroids[track_id] = cy

        else:
            # No detections this frame: still draw the counting line so it's visible
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)

        # write annotated frame to the output video
        writer.write(frame)

    # cleanup
    cap.release()
    writer.release()

    # convert defaultdict to regular dict for return (nicer to consume)
    return out_path, dict(class_counts)


def process_video_stream(
    video_path,
    model: YOLO,
    classes=None,
    conf_thres=0.25,
    line_position=0.75,
    frame_update_interval=5,
    on_progress=None,
):
    """
    Stream-processing version of process_video that calls `on_progress` periodically.

    Parameters
    - video_path: input video file path
    - model: loaded YOLO model
    - classes: list of class indices to keep (None = all)
    - conf_thres: detection confidence threshold
    - line_position: relative vertical position for counting (0..1 measured from top)
    - frame_update_interval: call on_progress every N frames (1 => every frame)
    - on_progress: callable(frame_rgb, counts_dict, progress_fraction) -> None
        - frame_rgb: current annotated frame as RGB numpy array (H,W,3) for display
        - counts_dict: current counts dict (class -> count)
        - progress_fraction: float between 0..1 indicating progress through video

    Returns
    - (annotated_video_path, counts_dict) same as process_video
    """

    import cv2
    import tempfile
    import os
    from collections import defaultdict

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    # output file (temporary)
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    line_y = int(height * line_position)
    class_counts = defaultdict(int)
    crossed_ids = set()
    last_centroids = dict()

    frame_idx = 0

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking/detection on frame
        results = model.track(frame, persist=True, conf=conf_thres, classes=classes)

        boxes = []
        track_ids = []
        class_indices = []
        confidences = []
        names = model.names if hasattr(model, "names") else {}

        # Guard against empty results
        if results and len(results) > 0 and getattr(results[0].boxes, "data", None) is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            # draw counting line
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)

            for (x1_f, y1_f, x2_f, y2_f), track_id, cls_idx, conf in zip(
                boxes, track_ids, class_indices, confidences
            ):
                x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cls_name = names[int(cls_idx)] if int(cls_idx) in names else str(int(cls_idx))

                # drawing
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id} {cls_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # counting logic using centroid transition
                prev_cy = last_centroids.get(track_id, None)
                if prev_cy is not None:
                    if prev_cy <= line_y and cy > line_y and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        class_counts[cls_name] += 1
                # update last centroid
                last_centroids[track_id] = cy
        else:
            # draw counting line even if no detections
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)

        # Write annotated frame to output
        writer.write(frame)

        # Send progress updates occasionally to avoid too many UI updates
        if on_progress is not None and (frame_idx % frame_update_interval == 0):
            # Convert BGR -> RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # create a shallow copy of counts to avoid mutation issues in UI
            counts_snapshot = dict(class_counts)
            # progress fraction (if total_frames is known)
            progress_frac = None
            if total_frames and total_frames > 0:
                progress_frac = min(1.0, frame_idx / float(total_frames))
            # call user callback
            try:
                on_progress(frame_rgb, counts_snapshot, progress_frac)
            except Exception:
                # callbacks should never break processing; ignore exceptions from UI callback
                pass

        frame_idx += 1

    # cleanup
    cap.release()
    writer.release()

    return out_path, dict(class_counts)