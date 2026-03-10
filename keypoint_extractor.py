import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

Holistic = mp.solutions.holistic.Holistic
PoseLandmark = mp.solutions.holistic.PoseLandmark
HandLandmark = mp.solutions.hands.HandLandmark

POSE_PARENTS = {
    PoseLandmark.NOSE: PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE_INNER: PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE: PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE_OUTER: PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EYE_INNER: PoseLandmark.NOSE,
    PoseLandmark.RIGHT_EYE: PoseLandmark.NOSE,
    PoseLandmark.RIGHT_EYE_OUTER: PoseLandmark.RIGHT_EYE,
    PoseLandmark.LEFT_EAR: PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EAR: PoseLandmark.RIGHT_EYE,
    PoseLandmark.MOUTH_LEFT: PoseLandmark.NOSE,
    PoseLandmark.MOUTH_RIGHT: PoseLandmark.NOSE,
    PoseLandmark.LEFT_SHOULDER: PoseLandmark.NOSE,
    PoseLandmark.RIGHT_SHOULDER: PoseLandmark.NOSE,
    PoseLandmark.LEFT_ELBOW: PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW: PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_WRIST: PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_WRIST: PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_PINKY: PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_PINKY: PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_INDEX: PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_INDEX: PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_THUMB: PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_THUMB: PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP: PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_HIP: PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_KNEE: PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_KNEE: PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_ANKLE: PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_ANKLE: PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_HEEL: PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_HEEL: PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.LEFT_FOOT_INDEX: PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_FOOT_INDEX: PoseLandmark.RIGHT_ANKLE,
}

# Basic mapping for hand joints to build transforms inside each hand.
HAND_PARENTS = {
    HandLandmark.WRIST: HandLandmark.WRIST,
    HandLandmark.THUMB_CMC: HandLandmark.WRIST,
    HandLandmark.THUMB_MCP: HandLandmark.THUMB_CMC,
    HandLandmark.THUMB_IP: HandLandmark.THUMB_MCP,
    HandLandmark.THUMB_TIP: HandLandmark.THUMB_IP,
    HandLandmark.INDEX_FINGER_MCP: HandLandmark.WRIST,
    HandLandmark.INDEX_FINGER_PIP: HandLandmark.INDEX_FINGER_MCP,
    HandLandmark.INDEX_FINGER_DIP: HandLandmark.INDEX_FINGER_PIP,
    HandLandmark.INDEX_FINGER_TIP: HandLandmark.INDEX_FINGER_DIP,
    HandLandmark.MIDDLE_FINGER_MCP: HandLandmark.WRIST,
    HandLandmark.MIDDLE_FINGER_PIP: HandLandmark.MIDDLE_FINGER_MCP,
    HandLandmark.MIDDLE_FINGER_DIP: HandLandmark.MIDDLE_FINGER_PIP,
    HandLandmark.MIDDLE_FINGER_TIP: HandLandmark.MIDDLE_FINGER_DIP,
    HandLandmark.RING_FINGER_MCP: HandLandmark.WRIST,
    HandLandmark.RING_FINGER_PIP: HandLandmark.RING_FINGER_MCP,
    HandLandmark.RING_FINGER_DIP: HandLandmark.RING_FINGER_PIP,
    HandLandmark.RING_FINGER_TIP: HandLandmark.RING_FINGER_DIP,
    HandLandmark.PINKY_MCP: HandLandmark.WRIST,
    HandLandmark.PINKY_PIP: HandLandmark.PINKY_MCP,
    HandLandmark.PINKY_DIP: HandLandmark.PINKY_PIP,
    HandLandmark.PINKY_TIP: HandLandmark.PINKY_DIP,
}

# Mapping from Mediapipe pose landmarks to avatar skeleton node names.
POSE_TO_AVATAR = {
    PoseLandmark.NOSE: "Head",
    PoseLandmark.LEFT_EYE: "LeftEye",
    PoseLandmark.RIGHT_EYE: "RightEye",
    PoseLandmark.LEFT_SHOULDER: "LeftShoulder",
    PoseLandmark.RIGHT_SHOULDER: "RightShoulder",
    PoseLandmark.LEFT_ELBOW: "LeftArm",
    PoseLandmark.RIGHT_ELBOW: "RightArm",
    PoseLandmark.LEFT_WRIST: "LeftForeArm",
    PoseLandmark.RIGHT_WRIST: "RightForeArm",
    PoseLandmark.LEFT_HIP: "LeftUpLeg",
    PoseLandmark.RIGHT_HIP: "RightUpLeg",
    PoseLandmark.LEFT_KNEE: "LeftLeg",
    PoseLandmark.RIGHT_KNEE: "RightLeg",
    PoseLandmark.LEFT_ANKLE: "LeftFoot",
    PoseLandmark.RIGHT_ANKLE: "RightFoot",
    PoseLandmark.LEFT_FOOT_INDEX: "LeftToeBase",
    PoseLandmark.RIGHT_FOOT_INDEX: "RightToeBase",
}

LEFT_HAND_TO_AVATAR = {
    HandLandmark.WRIST: "LeftHand",
    HandLandmark.THUMB_CMC: "LeftHandThumb1",
    HandLandmark.THUMB_MCP: "LeftHandThumb2",
    HandLandmark.THUMB_IP: "LeftHandThumb3",
    HandLandmark.THUMB_TIP: "LeftHandThumb4",
    HandLandmark.INDEX_FINGER_MCP: "LeftHandIndex1",
    HandLandmark.INDEX_FINGER_PIP: "LeftHandIndex2",
    HandLandmark.INDEX_FINGER_DIP: "LeftHandIndex3",
    HandLandmark.INDEX_FINGER_TIP: "LeftHandIndex4",
    HandLandmark.MIDDLE_FINGER_MCP: "LeftHandMiddle1",
    HandLandmark.MIDDLE_FINGER_PIP: "LeftHandMiddle2",
    HandLandmark.MIDDLE_FINGER_DIP: "LeftHandMiddle3",
    HandLandmark.MIDDLE_FINGER_TIP: "LeftHandMiddle4",
    HandLandmark.RING_FINGER_MCP: "LeftHandRing1",
    HandLandmark.RING_FINGER_PIP: "LeftHandRing2",
    HandLandmark.RING_FINGER_DIP: "LeftHandRing3",
    HandLandmark.RING_FINGER_TIP: "LeftHandRing4",
    HandLandmark.PINKY_MCP: "LeftHandPinky1",
    HandLandmark.PINKY_PIP: "LeftHandPinky2",
    HandLandmark.PINKY_DIP: "LeftHandPinky3",
    HandLandmark.PINKY_TIP: "LeftHandPinky4",
}

RIGHT_HAND_TO_AVATAR = {
    HandLandmark.WRIST: "RightHand",
    HandLandmark.THUMB_CMC: "RightHandThumb1",
    HandLandmark.THUMB_MCP: "RightHandThumb2",
    HandLandmark.THUMB_IP: "RightHandThumb3",
    HandLandmark.THUMB_TIP: "RightHandThumb4",
    HandLandmark.INDEX_FINGER_MCP: "RightHandIndex1",
    HandLandmark.INDEX_FINGER_PIP: "RightHandIndex2",
    HandLandmark.INDEX_FINGER_DIP: "RightHandIndex3",
    HandLandmark.INDEX_FINGER_TIP: "RightHandIndex4",
    HandLandmark.MIDDLE_FINGER_MCP: "RightHandMiddle1",
    HandLandmark.MIDDLE_FINGER_PIP: "RightHandMiddle2",
    HandLandmark.MIDDLE_FINGER_DIP: "RightHandMiddle3",
    HandLandmark.MIDDLE_FINGER_TIP: "RightHandMiddle4",
    HandLandmark.RING_FINGER_MCP: "RightHandRing1",
    HandLandmark.RING_FINGER_PIP: "RightHandRing2",
    HandLandmark.RING_FINGER_DIP: "RightHandRing3",
    HandLandmark.RING_FINGER_TIP: "RightHandRing4",
    HandLandmark.PINKY_MCP: "RightHandPinky1",
    HandLandmark.PINKY_PIP: "RightHandPinky2",
    HandLandmark.PINKY_DIP: "RightHandPinky3",
    HandLandmark.PINKY_TIP: "RightHandPinky4",
}


@dataclass
class JointSample:
    xyz: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _vector_to_quaternion(vec: np.ndarray) -> Tuple[float, float, float, float]:
    """Derive a quaternion that rotates +Y into vec."""
    target = _unit(vec)
    if np.linalg.norm(target) == 0:
        return (0.0, 0.0, 0.0, 1.0)

    source = np.array([0.0, 1.0, 0.0])
    half = _unit(source + target)
    if np.linalg.norm(half) == 0:
        # Opposite vector, rotate 180 around arbitrary perpendicular axis.
        axis = _unit(np.cross(source, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) == 0:
            axis = np.array([0.0, 0.0, 1.0])
        return (axis[0], axis[1], axis[2], 0.0)

    w = np.dot(source, half)
    xyz = np.cross(source, half)
    return (xyz[0], xyz[1], xyz[2], w)


def _landmarks_to_samples(
    landmarks: Iterable[landmark_pb2.Landmark],
    parents: Dict[int, int],
    names: Iterable[Optional[str]],
) -> Dict[str, JointSample]:
    samples = {}
    for idx, (name, point) in enumerate(zip(names, landmarks)):
        if not name:
            continue
        parent_idx = parents.get(idx, idx)
        parent_point = landmarks[parent_idx]
        vec = np.array(
            [
                point.x - parent_point.x,
                point.y - parent_point.y,
                point.z - parent_point.z,
            ]
        )
        samples[name] = JointSample(
            xyz=(point.x, point.y, point.z),
            quat=_vector_to_quaternion(vec),
        )
    return samples


def _pose_joint_names() -> List[str]:
    return [POSE_TO_AVATAR.get(landmark) for landmark in PoseLandmark]


def _hand_joint_names(prefix: str) -> List[Optional[str]]:
    mapping = LEFT_HAND_TO_AVATAR if prefix == "LEFT" else RIGHT_HAND_TO_AVATAR
    return [mapping.get(landmark) for landmark in HandLandmark]


def extract_keypoints(
    video_paths: List[str],
    output_dir: str,
    fps: int = 30,
) -> List[str]:
    """
    Extract xyz + rotation quaternions for each frame of the provided videos.
    Returns list of JSON files written to output_dir (one per video).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_paths: List[str] = []

    with Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video for keypoints: {video_path}")
                continue

            base = Path(video_path).stem
            json_path = os.path.join(
                output_dir,
                f"{base}_keypoints.json",
            )
            frames_payload = []
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            effective_fps = fps if fps > 0 else (cap_fps if cap_fps > 0 else 30.0)

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)

                if result.pose_world_landmarks:
                    pose_sample = _landmarks_to_samples(
                        result.pose_world_landmarks.landmark,
                        {
                            landmark.value: POSE_PARENTS[landmark].value
                            for landmark in PoseLandmark
                        },
                        _pose_joint_names(),
                    )
                else:
                    pose_sample = {}

                left_hand_sample = (
                    _landmarks_to_samples(
                        result.left_hand_landmarks.landmark,
                        {lm.value: HAND_PARENTS[lm].value for lm in HandLandmark},
                        _hand_joint_names("LEFT"),
                    )
                    if result.left_hand_landmarks
                    else {}
                )
                right_hand_sample = (
                    _landmarks_to_samples(
                        result.right_hand_landmarks.landmark,
                        {lm.value: HAND_PARENTS[lm].value for lm in HandLandmark},
                        _hand_joint_names("RIGHT"),
                    )
                    if result.right_hand_landmarks
                    else {}
                )

                frame_payload = {
                    "frame": frame_index,
                    "time": frame_index / effective_fps,
                    "joints": {
                        **{name: sample.__dict__ for name, sample in pose_sample.items()},
                        **{
                            name: sample.__dict__
                            for name, sample in left_hand_sample.items()
                        },
                        **{
                            name: sample.__dict__
                            for name, sample in right_hand_sample.items()
                        },
                    },
                }
                frames_payload.append(frame_payload)
                frame_index += 1

            cap.release()

            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(frames_payload, json_file, indent=2)

            json_paths.append(json_path)
            print(f"[INFO] Keypoints saved to {json_path}")

    return json_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Mediapipe keypoints and map them to avatar joints."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="One or more input video files to process.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where *_keypoints.json files will be written.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Optional FPS override used when computing timestamps.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extract_keypoints(
        video_paths=args.videos,
        output_dir=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
