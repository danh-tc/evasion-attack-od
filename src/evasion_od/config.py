"""Paths, model zoo registry and attack hyperparameter defaults.

Checkpoint/config filenames below must match exactly what `setup_env.sh`
downloads via `mim download mmdet --config <name> --dest checkpoints/`, since
`mim download` places both `<name>.py` (fully resolved config) and the
`.pth` checkpoint side by side in that directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data" / "coco"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "results"

COCO_IMAGES_DIR = DATA_DIR / "val2017"
COCO_ANNOTATIONS = DATA_DIR / "annotations" / "instances_val2017.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    group: str  # "surrogate" | "A" | "B" | "C" | "aux"
    config_name: str  # mim --config value, also the stem of the downloaded .py file
    checkpoint_file: str
    backbone: str
    paradigm: str

    @property
    def config_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.config_name}.py"

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / self.checkpoint_file


SURROGATE_NAME = "faster_rcnn_r50_fpn"

MODEL_ZOO: dict[str, ModelSpec] = {
    SURROGATE_NAME: ModelSpec(
        name=SURROGATE_NAME,
        group="surrogate",
        config_name="faster-rcnn_r50_fpn_1x_coco",
        checkpoint_file="faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        backbone="ResNet-50",
        paradigm="two-stage",
    ),
    # Group A -- in-family (ResNet-50 backbone), different paradigm
    "fcos_r50": ModelSpec(
        name="fcos_r50",
        group="A",
        config_name="fcos_r50-caffe_fpn_gn-head_1x_coco",
        checkpoint_file="fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
        backbone="ResNet-50",
        paradigm="anchor-free",
    ),
    "deformable_detr": ModelSpec(
        name="deformable_detr",
        group="A",
        config_name="deformable-detr_r50_16xb2-50e_coco",
        checkpoint_file="deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth",
        backbone="ResNet-50",
        paradigm="transformer",
    ),
    # Group B -- near-family, non-ResNet CNN backbone
    "yolov3_d53": ModelSpec(
        name="yolov3_d53",
        group="B",
        config_name="yolov3_d53_mstrain-608_273e_coco",
        checkpoint_file="yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
        backbone="Darknet-53",
        paradigm="anchor",
    ),
    "yolox_l": ModelSpec(
        name="yolox_l",
        group="B",
        config_name="yolox_l_8x8_300e_coco",
        checkpoint_file="yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        backbone="CSPNet",
        paradigm="anchor-free",
    ),
    # Group C -- cross-family, Swin Transformer backbone
    "mask_rcnn_swin_t": ModelSpec(
        name="mask_rcnn_swin_t",
        group="C",
        config_name="mask-rcnn_swin-t-p4-w7_fpn_1x_coco",
        checkpoint_file="mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
        backbone="Swin-T",
        paradigm="two-stage",
    ),
    "dino_swin_l": ModelSpec(
        name="dino_swin_l",
        group="C",
        config_name="dino-5scale_swin-l_8xb2-12e_coco",
        checkpoint_file="dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth",
        backbone="Swin-L",
        paradigm="full-transformer",
    ),
    # Auxiliary / ablation models, not part of the main cross-family table
    "retinanet_r50": ModelSpec(
        name="retinanet_r50",
        group="aux",
        config_name="retinanet_r50_fpn_1x_coco",
        checkpoint_file="retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
        backbone="ResNet-50",
        paradigm="anchor",
    ),
    "retinanet_r101": ModelSpec(
        name="retinanet_r101",
        group="aux",
        config_name="retinanet_r101_fpn_1x_coco",
        checkpoint_file="retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth",
        backbone="ResNet-101",
        paradigm="anchor",
    ),
    "dino_r50": ModelSpec(
        name="dino_r50",
        group="aux",
        config_name="dino-4scale_r50_8xb2-12e_coco",
        checkpoint_file="dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth",
        backbone="ResNet-50",
        paradigm="full-transformer",
    ),
}

# Cross-family target list used by the main experiments (excludes surrogate/aux)
TARGET_NAMES: list[str] = [
    name for name, spec in MODEL_ZOO.items() if spec.group in ("A", "B", "C")
]

# Two representative targets used during the cheap Pilot Study sweep
PILOT_TARGET_NAMES: list[str] = ["fcos_r50", "mask_rcnn_swin_t"]


@dataclass
class AttackConfig:
    """Hyperparameters for one MI-FGSM + backbone-feature attack run.

    Pixel values are kept in raw [0, 255] scale throughout (matches OSFD's
    convention), so epsilon/alpha are expressed in that same scale.
    """

    epsilon: float = 5.0
    alpha: float = 1.0
    max_iterations: int = 100
    momentum: float = 1.0

    # Loss: L = sum_i mean_j (F_adv_ij - k * F_clean_ij)^2 ; k=1 -> NRDM, k=3 -> OSFD
    k: float = 3.0

    # RaPA DropConnect masking, scoped to backbone only
    mask_enabled: bool = False
    drop_prob: float = 0.05
    mask_layer_types: tuple[str, ...] = ("BatchNorm2d",)
    num_masks: int = 1  # "S" in RaPA paper / "--masks" in run_sweep.py

    # OSFD's RRB augmentation (rotation + resize + blur) -- off by default (E1/E3/E4/E5)
    use_rrb: bool = False

    seed: int = 42


@dataclass
class ExperimentSpec:
    """One named row of the E1..E5 comparison table in plan.md."""

    name: str
    attack: AttackConfig = field(default_factory=AttackConfig)


def make_experiments(drop_prob: float, num_masks: int) -> dict[str, ExperimentSpec]:
    """Build E1..E5 given a (p*, S*) chosen from the Pilot Study."""
    return {
        "E1_osfd_baseline": ExperimentSpec(
            "E1_osfd_baseline", AttackConfig(k=3.0, mask_enabled=False, use_rrb=False)
        ),
        "E2_osfd_rrb": ExperimentSpec(
            "E2_osfd_rrb", AttackConfig(k=3.0, mask_enabled=False, use_rrb=True)
        ),
        "E3_nrdm_control": ExperimentSpec(
            "E3_nrdm_control", AttackConfig(k=1.0, mask_enabled=False, use_rrb=False)
        ),
        "E4_rapa_od_baseline": ExperimentSpec(
            "E4_rapa_od_baseline",
            AttackConfig(
                k=1.0,
                mask_enabled=True,
                drop_prob=drop_prob,
                num_masks=num_masks,
                use_rrb=False,
            ),
        ),
        "E5_rapa_od_osfd_loss": ExperimentSpec(
            "E5_rapa_od_osfd_loss",
            AttackConfig(
                k=3.0,
                mask_enabled=True,
                drop_prob=drop_prob,
                num_masks=num_masks,
                use_rrb=False,
            ),
        ),
        "I4_rapa_od_rrb": ExperimentSpec(
            "I4_rapa_od_rrb",
            AttackConfig(
                k=3.0,
                mask_enabled=True,
                drop_prob=drop_prob,
                num_masks=num_masks,
                use_rrb=True,
            ),
        ),
    }
