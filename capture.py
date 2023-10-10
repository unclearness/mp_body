# https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from pathlib import Path
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


def main():
    if not MP_TASK_FILE_PATH.exists():
        import requests
        import os
        urlData = requests.get(MP_TASK_FILE_URL).content
        os.makedirs(MP_TASK_FILE_PATH.parent, exist_ok=True)
        with open(MP_TASK_FILE_PATH, mode='wb') as fp:
            fp.write(urlData)

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path=MP_TASK_FILE_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        st = time.time()
        detection_result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
        et = time.time()
        print("{:.2f}".format((et - st) * 1000), "ms")
        print("\033[1A", end="")

        if detection_result.segmentation_masks is None:
            continue

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(frame,
                                                  detection_result)
        cv2.imshow("annotated", annotated_image)
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(
            segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        cv2.imshow("mask", visualized_mask)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
