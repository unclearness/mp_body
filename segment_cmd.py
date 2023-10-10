# https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import cv2
import argparse
import time


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                             z=landmark.z)
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


MP_TASK_FILE_URL_LITE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MP_TASK_FILE_URL_HEAVY = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MP_TASK_FILE_URL = MP_TASK_FILE_URL_LITE  # MP_TASK_FILE_URL_HEAVY
MP_TASK_FILE_NAME = Path(MP_TASK_FILE_URL).name
MP_TASK_FILE_PATH = Path(__file__).parent.joinpath("data", MP_TASK_FILE_NAME)


def main(path):
    if not MP_TASK_FILE_PATH.exists():
        import requests
        import os
        urlData = requests.get(MP_TASK_FILE_URL).content
        os.makedirs(MP_TASK_FILE_PATH.parent, exist_ok=True)
        with open(MP_TASK_FILE_PATH, mode='wb') as fp:
            fp.write(urlData)

    basename = Path(path).name

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path=MP_TASK_FILE_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(path)

    # STEP 4: Detect pose landmarks from the input image.
    st = time.time()
    detection_result = detector.detect(image)
    et = time.time()
    print("{:.2f}".format((et - st) * 1000), "ms")

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(
        image.numpy_view()[..., :3], detection_result)
    cv2.imwrite("annotated_" + basename,
                cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(
        segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imwrite("mask_" + basename, visualized_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('path', help='')

    args = parser.parse_args()
    main(args.path)
