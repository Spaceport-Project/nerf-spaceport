"""Data parser for Spaceport in-house multicamera setup dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import cv2
import imageio
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import (CAMERA_MODEL_TO_TYPE, Cameras,
                                        CameraType)
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


@dataclass
class NerfplayerMulticamDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: NerfplayerMulticam)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
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

    def _process_frames(self, frame_names: List[str], time_ids: np.ndarray) -> Tuple[List, List]:
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
            cam_json = load_from_json(self.data / f"camera/{frame}.json")

            if "transform_matrix" in cam_json:
                trans_mat = cam_json["transform_matrix"]

                # Assert that trans_mat is a list or a list of lists with numeric values
                assert isinstance(trans_mat, list), "trans_mat must be a list"
                assert all(isinstance(row, list)
                           for row in trans_mat), "Each element of trans_mat must be a list"
                assert all(all(isinstance(el, (int, float)) for el in row)
                           for row in trans_mat), "Each element in the sublists of trans_mat must be a number"

                trans_mat_np = np.array(trans_mat)

                # Assert that trans_mat_np can be sliced into shape [:3, :4]
                assert trans_mat_np.shape[0] >= 3 and trans_mat_np.shape[
                    1] >= 4, "trans_mat_np must be at least of shape (3, 4)"

                pose = trans_mat_np[:3, :4]

            if "fl_x" in cam_json:
                fx_fixed = cam_json["fl_x"]

                # Assert that fx_fixed is a number
                assert isinstance(fx_fixed, (int, float)
                                  ), "fx_fixed must be a number"

            if "fl_y" in cam_json:
                fy_fixed = cam_json["fl_y"]

                # Assert that fy_fixed is a number
                assert isinstance(fy_fixed, (int, float)
                                  ), "fy_fixed must be a number"

            if "c_x" in cam_json:
                cx_fixed = cam_json["c_x"]

                # Assert that cx_fixed is a number
                assert isinstance(cx_fixed, (int, float)
                                  ), "cx_fixed must be a number"

            if "c_y" in cam_json:
                cy_fixed = cam_json["c_y"]

                # Assert that cy_fixed is a number
                assert isinstance(cy_fixed, (int, float)
                                  ), "cy_fixed must be a number"

            if "h" in cam_json:
                height_fixed = cam_json["h"]

                # Assert that height_fixed is an integer
                assert isinstance(
                    height_fixed, int), "height_fixed must be an integer"

            if "w" in cam_json:
                width_fixed = cam_json["w"]

                # Assert that width_fixed is an integer
                assert isinstance(
                    width_fixed, int), "width_fixed must be an integer"

            distort = []
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(cam_json["k1"]) if "k1" in cam_json else 0.0,
                    k2=float(cam_json["k2"]) if "k2" in cam_json else 0.0,
                    k3=float(cam_json["k3"]) if "k3" in cam_json else 0.0,
                    k4=float(cam_json["k4"]) if "k4" in cam_json else 0.0,
                    p1=float(cam_json["p1"]) if "p1" in cam_json else 0.0,
                    p2=float(cam_json["p2"]) if "p2" in cam_json else 0.0,
                )
            )

            if "camera_model" in cam_json:
                camera_type = CAMERA_MODEL_TO_TYPE[cam_json["camera_model"]]
            else:
                camera_type = CameraType.PERSPECTIVE

            cams.append(
                {
                    "camera_to_worlds": pose,
                    "fx": fx_fixed,
                    "fy": fy_fixed,
                    "cx": cx_fixed,
                    "cy": cy_fixed,
                    "height": height_fixed,
                    "width": width_fixed,
                    "distortion_params": distort,
                    "times": torch.as_tensor(time_ids[idx] / self._time_ids.max()).float(),
                }
            )

        d = self.config.downscale_factor
        if not image_filenames[0].exists():
            CONSOLE.print(f"downscale factor {d}x not exist, converting")
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

        return image_filenames, cams

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

        image_filenames, cams = self._process_frames(
            frame_names.tolist(), time_ids)

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale],
                                                [aabb_scale, aabb_scale, aabb_scale]],
                                               dtype=torch.float32))

        cam_dict = {}

        """for k in cams[0].keys():
            print(f"Key: {k}, Value: {cams[0][k]}, Type: {type(cams[0][k])}")

            # Handling list of tensors by extracting the tensor before stacking
            if isinstance(cams[0][k], list) and isinstance(cams[0][k][0], torch.Tensor):
                cam_dict[k] = torch.stack([c[k][0].float() for c in cams])

            # Handling 'height' and 'width' by converting them to integer tensors
            elif k in ['height', 'width']:
                cam_dict[k] = torch.stack(
                    [torch.tensor(c[k], dtype=torch.int32) for c in cams])

            # Handling other cases by converting them to float tensors
            else:
                cam_dict[k] = torch.stack(
                    [torch.tensor(c[k], dtype=torch.float32) for c in cams])"""

        for k in cams[0].keys():
            # print(f"Key: {k}, Value: {cams[0][k]}, Type: {type(cams[0][k])}")

            # Explicit handling for 'distort' key
            if k == 'distortion_params':
                cam_dict[k] = torch.stack([c[k][0].float() for c in cams])

            # Handling 'height' and 'width' by converting them to integer tensors
            elif k in ['height', 'width']:
                cam_dict[k] = torch.stack(
                    [torch.tensor(c[k], dtype=torch.int32) for c in cams])

            # Handling other cases by converting them to float tensors
            else:
                # Verify that the data is not a boolean before converting to float tensor
                if not isinstance(cams[0][k], bool):
                    cam_dict[k] = torch.stack(
                        [torch.tensor(c[k], dtype=torch.float32) for c in cams])
                else:
                    print(f"Warning: Skipping key '{k}' due to boolean value.")

        cameras = Cameras(camera_type=CameraType.PERSPECTIVE, **cam_dict)

        dataparser_outputs = DataparserOutputs(image_filenames=image_filenames,
                                               cameras=cameras,
                                               scene_box=scene_box,
                                               alpha_color=alpha_color_tensor,
                                               )

        return dataparser_outputs
