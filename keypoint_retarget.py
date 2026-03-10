import argparse
import json
from base64 import b64encode
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from pygltflib import (
    FLOAT,
    SCALAR,
    VEC3,
    VEC4,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Buffer,
    BufferView,
    GLTF2,
)

DEFAULT_ROTATION = (0.0, 0.0, 0.0, 1.0)
DEFAULT_TRANSLATION = (0.0, 0.0, 0.0)
FLIP_Z_MATRIX = np.diag([1.0, 1.0, -1.0])  # Mediapipe (Z forward) -> glTF (Z backward)

# A conservative list of joints that we animate by default.
# This avoids driving every single joint (including hips/feet),
# which can look very unstable when Mediapipe data is noisy or
# when the avatar's rest pose / rig differs from the capture pose.
DEFAULT_ANIMATED_JOINTS = {
    # Head / neck
    "Head",
    "LeftEye",
    "RightEye",
    # Shoulders / arms
    "LeftShoulder",
    "RightShoulder",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    # Hands (wrists)
    "LeftHand",
    "RightHand",
}


def _quaternion_multiply(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Multiply two quaternions (xyzw format)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quaternion_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Return the conjugate of a quaternion (xyzw format)."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_to_matrix(quat: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion (xyzw) to 3x3 rotation matrix."""
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (xyzw)."""
    m = matrix
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def _transform_mediapipe_to_gltf_quaternion(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert a MediaPipe quaternion into glTF's coordinate space by flipping the Z axis.
    """
    rot_matrix = _quat_to_matrix(quat)
    transformed_matrix = FLIP_Z_MATRIX @ rot_matrix @ FLIP_Z_MATRIX
    return _matrix_to_quat(transformed_matrix)


def _normalize_quaternion(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Normalize a quaternion to unit length."""
    x, y, z, w = q
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/norm, y/norm, z/norm, w/norm)


def _load_frames(json_path: Path) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as fp:
        frames = json.load(fp)

    if not frames:
        raise ValueError(f"No frames found in {json_path}")
    return frames


def _load_mapping(mapping_path: Optional[Path], joint_names: Iterable[str]) -> Dict[str, str]:
    mapping = {name: name for name in joint_names}
    if not mapping_path:
        return mapping

    with open(mapping_path, "r", encoding="utf-8") as fp:
        user_mapping = json.load(fp)

    mapping.update(user_mapping)
    return mapping


def _derive_times(frames: List[Dict], fps: float) -> np.ndarray:
    times: List[float] = []
    for idx, frame in enumerate(frames):
        if "time" in frame:
            times.append(float(frame["time"]))
        elif fps > 0:
            times.append(idx / fps)
        else:
            raise ValueError("Keyframe time missing and fps not provided (>0).")

    return np.array(times, dtype=np.float32)


def _gather_tracks(frames: List[Dict], joint_name: str, apply_coordinate_transform: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    translations: List[Tuple[float, float, float]] = []
    rotations: List[Tuple[float, float, float, float]] = []
    
    for frame in frames:
        payload = frame.get("joints", {}).get(joint_name)
        if not payload:
            translations.append(translations[-1] if translations else DEFAULT_TRANSLATION)
            rotations.append(rotations[-1] if rotations else DEFAULT_ROTATION)
            continue

        translations.append(tuple(payload.get("xyz", DEFAULT_TRANSLATION)))
        quat = tuple(payload.get("quat", DEFAULT_ROTATION))
        quat = _normalize_quaternion(quat)
        
        # Apply coordinate system transformation if needed
        if apply_coordinate_transform:
            quat = _transform_mediapipe_to_gltf_quaternion(quat)
            quat = _normalize_quaternion(quat)
        
        rotations.append(quat)

    return (np.array(translations, dtype=np.float32), np.array(rotations, dtype=np.float32))


def _add_buffer_blob(bytes_blob: bytearray, array: np.ndarray) -> Tuple[int, int]:
    offset = len(bytes_blob)
    bytes_blob.extend(array.tobytes())
    return offset, array.nbytes


def _ensure_list(container, default=None):
    if container is None:
        return default if default is not None else []
    return container


def retarget(
    gltf_path: Path,
    keypoints_path: Path,
    mapping_path: Optional[Path],
    output_path: Path,
    animation_name: str,
    fps: float,
) -> None:
    gltf = GLTF2().load(str(gltf_path))
    gltf.accessors = _ensure_list(gltf.accessors)
    gltf.bufferViews = _ensure_list(gltf.bufferViews)
    gltf.buffers = _ensure_list(gltf.buffers)
    gltf.animations = _ensure_list(gltf.animations)

    frames = _load_frames(keypoints_path)

    # Collect union of all joint names across frames, since the first frame
    # may have no detections.
    joint_name_set = set()
    for frame in frames:
        joint_name_set.update(frame.get("joints", {}).keys())
    joint_names = sorted(joint_name_set)
    if not joint_names:
        raise ValueError(f"No joints found in keypoints file: {keypoints_path}")
    mapping = _load_mapping(mapping_path, joint_names)

    # Restrict to a safer subset of joints unless the user provided
    # an explicit mapping file. This helps avoid extreme distortions
    # from joints whose orientations don't match the avatar rig well.
    if not mapping_path:
        mapping = {
            source_name: target_name
            for source_name, target_name in mapping.items()
            if source_name in DEFAULT_ANIMATED_JOINTS
        }

    node_indices = {node.name: idx for idx, node in enumerate(gltf.nodes or []) if node.name}
    missing_nodes = sorted({target for target in mapping.values() if target not in node_indices})
    if missing_nodes:
        raise ValueError(
            "The following target joints are missing in the glTF nodes: "
            + ", ".join(missing_nodes)
        )

    times = _derive_times(frames, fps)
    buffer_bytes = bytearray()
    buffers_offset_index = len(gltf.buffers)
    buffer_view_start = len(gltf.bufferViews)
    accessor_start = len(gltf.accessors)

    # Shared time accessor.
    time_offset, time_length = _add_buffer_blob(buffer_bytes, times)
    gltf.bufferViews.append(
        BufferView(
            buffer=buffers_offset_index,
            byteOffset=time_offset,
            byteLength=time_length,
        )
    )
    time_accessor_index = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            bufferView=len(gltf.bufferViews) - 1,
            componentType=FLOAT,
            count=len(times),
            type=SCALAR,
            min=[float(times.min())],
            max=[float(times.max())],
        )
    )

    animation = Animation(name=animation_name, samplers=[], channels=[])

    for source_name, target_name in mapping.items():
        # NOTE:
        # -----
        # The Mediapipe world-space translations in the keypoints JSON often do not
        # match the avatar's rig scale/origin, which can cause the avatar to
        # "explode" or drift far away when we directly write translation tracks.
        #
        # To keep the avatar stable and only drive its pose, we currently:
        #   * IGNORE translations from Mediapipe
        #   * ONLY write rotation tracks per joint
        #
        # If you ever want to reintroduce translations, you can restore the
        # previous translation-writing block and, ideally, normalize/offset the
        # coordinates to your avatar's root and scale.
        _translations, rotations = _gather_tracks(frames, source_name, apply_coordinate_transform=True)

        rot_offset, rot_length = _add_buffer_blob(buffer_bytes, rotations)
        gltf.bufferViews.append(
            BufferView(
                buffer=buffers_offset_index,
                byteOffset=rot_offset,
                byteLength=rot_length,
            )
        )
        rot_accessor_index = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=len(gltf.bufferViews) - 1,
                componentType=FLOAT,
                count=len(rotations),
                type=VEC4,
            )
        )

        # Rotation sampler / channel only (no translation).
        sampler_index = len(animation.samplers)
        animation.samplers.append(
            AnimationSampler(
                input=time_accessor_index,
                output=rot_accessor_index,
                interpolation="LINEAR",
            )
        )
        animation.channels.append(
            AnimationChannel(
                sampler=sampler_index,
                target=AnimationChannelTarget(
                    node=node_indices[target_name],
                    path="rotation",
                ),
            )
        )

    encoded_buffer = b64encode(buffer_bytes).decode("ascii")
    gltf.buffers.append(
        Buffer(
            byteLength=len(buffer_bytes),
            uri=f"data:application/octet-stream;base64,{encoded_buffer}",
        )
    )

    gltf.animations.append(animation)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".glb":
        gltf.save_binary(str(output_path))
    else:
        gltf.save(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed Mediapipe keypoint animation inside a glTF avatar."
    )
    parser.add_argument("--gltf", required=True, help="Path to the avatar glTF file.")
    parser.add_argument("--keypoints", required=True, help="Keypoints JSON exported by keypoint_extractor.py")
    parser.add_argument(
        "--mapping",
        help="Optional JSON file mapping keypoint names to glTF node names.",
    )
    parser.add_argument("--output", required=True, help="Output glTF or GLB with animation.")
    parser.add_argument(
        "--animation-name",
        default="MediapipeRetarget",
        help="Name of the animation clip stored in the glTF.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback frames-per-second if keypoints JSON lacks explicit times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retarget(
        gltf_path=Path(args.gltf),
        keypoints_path=Path(args.keypoints),
        mapping_path=Path(args.mapping) if args.mapping else None,
        output_path=Path(args.output),
        animation_name=args.animation_name,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

