from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from face_mesh.domain.entities import BodyViewEstimate
from face_mesh.domain.ports import BodyShapeEstimatorPort

_ROMP_PKL_URL = "https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl"


def _default_romp_pkl_path() -> Path:
    override = os.environ.get("ROMP_MODEL_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".romp" / "ROMP.pkl"


def _download_romp_checkpoint(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        _ROMP_PKL_URL,
        headers={"User-Agent": "face-mesh (urllib)"},
        method="GET",
    )
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as outfile:
            with urllib.request.urlopen(request, timeout=600) as response:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_romp_checkpoint_exists(path: Path | None = None) -> Path:
    target = path if path is not None else _default_romp_pkl_path()
    if not target.is_file():
        _download_romp_checkpoint(target)
    return target


class RompBodyShapeAdapter(BodyShapeEstimatorPort):
    def __init__(
        self,
        gpu_device: int | None = None,
        romp_model_path: Path | None = None,
    ) -> None:
        self._gpu_device = gpu_device
        self._romp_model_path = romp_model_path
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        import romp.main as romp_main

        ckpt = ensure_romp_checkpoint_exists(self._romp_model_path)
        gpu = self._gpu_device
        if gpu is None:
            gpu = 0 if torch.cuda.is_available() else -1
        settings = romp_main.romp_settings(
            [
                "--GPU",
                str(gpu),
                "--calc_smpl",
                "--model_path",
                str(ckpt),
            ]
        )
        self._model = romp_main.ROMP(settings)

    def estimate(self, image_bgr: Any) -> BodyViewEstimate:
        try:
            import numpy as np
        except ImportError:
            return BodyViewEstimate(
                betas=None,
                confidence=None,
                ok=False,
                message="numpy required for body estimation",
            )
        try:
            self._ensure_model()
        except Exception as exc:
            return BodyViewEstimate(
                betas=None,
                confidence=None,
                ok=False,
                message=str(exc),
            )
        assert self._model is not None
        outputs = self._model(image_bgr)
        if outputs is None:
            return BodyViewEstimate(
                betas=None,
                confidence=None,
                ok=False,
                message="No person detected",
            )
        betas_arr = outputs["smpl_betas"]
        cam = outputs["cam"]
        center_confs = outputs.get("center_confs")
        betas_arr = np.asarray(betas_arr, dtype=np.float64)
        if betas_arr.ndim == 1:
            betas_arr = betas_arr.reshape(1, -1)
        cam = np.asarray(cam, dtype=np.float64)
        if cam.ndim == 1:
            cam = cam.reshape(1, -1)
        idx = int(np.argmax(cam[:, 0]))
        betas_list = [round(float(x), 6) for x in betas_arr[idx].tolist()]
        confidence = None
        if center_confs is not None:
            cc = np.asarray(center_confs, dtype=np.float64).reshape(-1)
            if idx < len(cc):
                confidence = round(float(cc[idx]), 6)
        return BodyViewEstimate(
            betas=betas_list,
            confidence=confidence,
            ok=True,
            message=None,
        )
