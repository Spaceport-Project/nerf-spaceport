"""Data parser for Spaceport in-house multicamera setup dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (DataParser,
                                                         DataParserConfig,
                                                         DataparserOutputs)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


def _load_metadata_info(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Load scene metadata from json.

    Args:
        data_dir: data path

    Returns:
        A tuple of scene info: frame_names_map, time_ids, camera_ids
    """
    dataset_dict = load_from_json(data_dir / "dataset.json")
    _frame_names = np.array(dataset_dict["ids"])

    metadata_dict = load_from_json(data_dir / "metadata.json")
    time_ids = np.array([metadata_dict[k]["warp_id"]
                        for k in _frame_names], dtype=np.uint32)
    camera_ids = np.array([metadata_dict[k]["camera_id"]
                          for k in _frame_names], dtype=np.uint32)

    frame_names_map = np.zeros(
        (time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype)
    for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
        frame_names_map[t, c] = _frame_names[i]

    return frame_names_map, time_ids, camera_ids


def process_frames(self, frame_names: List[str], time_ids: np.ndarray, camera_ids: np.ndarray) -> Tuple[List, List]:
    """Read cameras and filenames from the name list.

    Args:
        frame_names: list of file names.
        time_ids: time id of each frame.

    Returns:
        A list of camera, each entry is a dict of the camera.
    """
    image_filenames = []
    poses = []
    cams = []
    for idx, frame in enumerate(frame_names):
        # print(f"rgb/{self.config.downscale_factor}x/{frame}.png")
        image_filenames.append(
            self.data / f"rgb/{self.config.downscale_factor}x/{frame}.png")
        poses.append(np.array(frame["transform_matrix"]))

        cam_json = load_from_json(self.data / f"camera/{frame}.json")
        c2w = torch.as_tensor(cam_json["orientation"]).T
        position = torch.as_tensor(cam_json["position"])
        position -= self._center  # some scenes look weird (wheel)
        position *= self._scale * self.config.scale_factor
        pose = torch.zeros([3, 4])
        pose[:3, :3] = c2w
        pose[:3, 3] = position
        # from opencv coord to opengl coord (used by nerfstudio)
        pose[0:3, 1:3] *= -1  # switch cam coord x,y
        pose = pose[[1, 0, 2], :]  # switch world x,y
        pose[2, :] *= -1  # invert world z
        # for aabb bbox usage
        pose = pose[[1, 2, 0], :]  # switch world xyz to zxy
        cams.append(
            {
                "camera_to_worlds": pose,
                "fx": cam_json["focal_length"] / self.config.downscale_factor,
                "fy": cam_json["focal_length"] * cam_json["pixel_aspect_ratio"] / self.config.downscale_factor,
                "cx": cam_json["principal_point"][0] / self.config.downscale_factor,
                "cy": cam_json["principal_point"][1] / self.config.downscale_factor,
                "height": cam_json["image_size"][1] // self.config.downscale_factor,
                "width": cam_json["image_size"][0] // self.config.downscale_factor,
                "times": torch.as_tensor(time_ids[idx] / self._time_ids.max()).float(),
            }
        )

    d = self.config.downscale_factor
    if not image_filenames[0].exists():
        CONSOLE.print(f"downscale factor {d}x not exist, converting")
        print(f"rgb/1x/{frame_names[0]}.png")
        ori_h, ori_w = cv2.imread(
            str(self.data / f"rgb/1x/{frame_names[0]}.png")).shape[:2]
        (self.data / f"rgb/{d}x").mkdir(exist_ok=True)
        h, w = ori_h // d, ori_w // d
        for frame in frame_names:
            cv2.imwrite(
                str(self.data / f"rgb/{d}x/{frame}.png"),
                cv2.resize(cv2.imread(
                    str(self.data / f"rgb/1x/{frame}.png")), (h, w)),
            )
        CONSOLE.print("finished")

    """if not depth_filenames[0].exists():
            CONSOLE.print(f"processed depth downscale factor {d}x not exist, converting")
            (self.data / f"processed_depth/{d}x").mkdir(exist_ok=True, parents=True)
            for idx, frame in enumerate(frame_names):
                depth = np.load(self.data / f"depth/1x/{frame}.npy")
                mask = rescale((depth != 0).astype(np.uint8) * 255, 1 / d, cv2.INTER_AREA)
                depth = rescale(depth, 1 / d, cv2.INTER_AREA)
                depth[mask != 255] = 0
                depth = _rescale_depth(depth, cams[idx])
                np.save(str(self.data / f"processed_depth/{d}x/{frame}.npy"), depth)
            CONSOLE.print("finished")"""
    depth_filenames = []
    return image_filenames, cams


@dataclass
class NerfplayerMulticamDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: NerfplayerMulticam)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: int = 1
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_box_bound: float = 1.5
    """Boundary of scene box."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class NerfplayerMulticam(DataParser):
    """Nerfplayer dataparser class for Spaceport in-house multicamera setup"""

    config: NerfplayerMulticamDataParserConfig
    includes_time: bool = True

    def __init__(self, config: NerfplayerMulticamDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self._frame_names_map, self._time_ids, self._camera_ids = _load_metadata_info(
            self.data)

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None
        splits_dir = self.data / "splits"

        if not (splits_dir / f"{split}.json").exists():
            CONSOLE.print(f"split {split} not found, using split train")
            split = "train"
        split_dict = load_from_json(splits_dir / f"{split}.json")
        frame_names = np.array(split_dict["frame_names"])
        time_ids = np.array(split_dict["time_ids"])
        camera_ids = np.array(split_dict["camera_ids"])

        if split != "train":
            CONSOLE.print(
                f"split {split} is empty, using the 1st training image")
            split_dict = load_from_json(splits_dir / "train.json")
            frame_names = np.array(split_dict["frame_names"])[[0]]
            time_ids = np.array(split_dict["time_ids"])[[0]]
            camera_ids = np.array(split_dict["camera_ids"])

        image_filenames, cams = self.process_frames(
            frame_names.tolist(), time_ids.tolist(), time_ids, camera_ids)

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
