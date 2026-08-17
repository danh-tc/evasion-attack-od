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

    # Loss: "osfd" -> L = sum_i mean_j (F_adv_ij - k*F_clean_ij)^2 (k=1 -> NRDM,
    # k=3 -> OSFD); "rel" -> relational contrast loss, standalone (idea.txt
    # Phase 2, see results/P2_rel_*_go.json -- NOGO, lost to E1 on 7/7
    # models); "osfd_rel_hybrid" -> OSFD + lambda * relational_diff
    # regularizer (Phase 2 follow-up, also NOGO, results/P2_hybrid_*_go.json);
    # "spatial" -> dense per-pixel object-to-context feature misalignment
    # (Phase 1b, results/P1b_prototype_diagnostic.json), see losses.py.
    loss_type: str = "osfd"
    k: float = 3.0

    # "rel"/"osfd_rel_hybrid" only: per-surrogate-backbone-stage weights (must
    # match the surrogate's stage count -- 4 for faster_rcnn_r50_fpn). e.g.
    # (0,0,0,1) attacks only the deepest stage, (.25,.25,.25,.25) all equally.
    rel_stage_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)
    rel_r: float = 1.0
    rel_min_margin_cells: int = 1
    # "osfd_rel_hybrid" only: weight on the relational_diff regularizer term.
    # Must be calibrated against L_OSFD's scale -- measured |L_OSFD|/|L_relD|
    # ~50-85x on a sample image, so lambda=O(1) makes it negligible; use
    # lambda~10-100 for it to meaningfully influence the gradient.
    rel_lambda: float = 30.0

    # "spatial" only: per-surrogate-backbone-stage weights for
    # spatial_misalignment_loss (same convention/stage-count as rel_stage_weights;
    # region construction reuses rel_r/rel_min_margin_cells above).
    spatial_stage_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    # RaPA DropConnect masking, scoped to backbone only
    mask_enabled: bool = False
    drop_prob: float = 0.05
    mask_layer_types: tuple[str, ...] = ("BatchNorm2d",)
    # Number of independent stochastic forward/backward passes averaged per
    # PGD step. "S" in the RaPA paper (independent masks) / "--masks" in
    # run_sweep.py; also doubles as SSA's "N" independent spectral copies
    # (new-plan.txt Sec 5.2.D) when augmentation="ssa", and as Phase G0's "K"
    # independent RRB views (plan.md, see grad_combine below) when
    # augmentation="rrb".
    num_masks: int = 1

    # Phase G0 (plan.md "RRB Gradient Mechanism Analysis") only: how the K
    # per-view gradients (K = num_masks) are combined into one MI-FGSM update
    # direction each iteration. "mean" is the pre-G0 behavior used by every
    # other experiment (E1-E9/I4/REL_*/HYBRID_*/SPATIAL_*). "consensus"/
    # "disagree" weight the mean gradient by (C_p)^gamma / (1-C_p)^gamma,
    # where C_p in [0,1] is the per-coordinate cross-view sign-agreement
    # (gradient_diagnostics.py); "consensus_shuffle" is the spatial-shuffle
    # control for "consensus" (same weight distribution, wrong positions).
    grad_combine: str = "mean"
    grad_combine_gamma: float = 1.0
    # Phase G0 causal comparisons only (e.g. RRB_K5_MEAN vs RRB_K5_CONS):
    # when True, reseeds Python's/PyTorch's RNG deterministically as a
    # function of (seed, image_id, iteration, k-index) right before each of
    # the K augmentation draws, so two runs that only differ in
    # grad_combine see the *same* K RRB views (rotation angle, resize scale,
    # noise) at every corresponding (image, iteration, k) -- "common random
    # numbers" variance reduction, isolating grad_combine as the only real
    # difference between the two trajectories instead of confounding it with
    # independently-sampled augmentation. Off by default (every non-G0
    # experiment, and the original G0 4-variant design, sampled freely).
    deterministic_augmentation: bool = False

    # Input-level augmentation applied to the adversarial image before the
    # surrogate forward pass. "none" (E1/E3/E4/E5) / "rrb" (OSFD's rotation +
    # resize + noise, E2/I4) / "dim" / "ssa" / "srs" (new-plan.txt Sec 5.2) /
    # "rrb_spectral" (Option A: spectral on top of unmodified RRB, E9) /
    # "rrb_rot"/"rrb_resize"/"rrb_noise"/"rrb_rot_resize" (Phase B1 component
    # ablation) / "fixed_scale" (Phase S0 scale sweep, uses `fixed_scale`
    # below instead of RRB's randomized resize).
    augmentation: str = "none"
    # "fixed_scale" only: deterministic global scale factor (Phase S0).
    fixed_scale: float = 1.0
    # "rrb_occupancy" only: random content-occupancy range for
    # `random_occupancy_resize`, replacing RRB's `adaptive_random_resizing`
    # (Phase S1 "Bidirectional/Shrink-aware RRB").
    occ_low: float = 0.7
    occ_high: float = 1.0

    seed: int = 42


@dataclass
class ExperimentSpec:
    """One named row of the E1..E9/I4 comparison table in plan.md."""

    name: str
    attack: AttackConfig = field(default_factory=AttackConfig)


def make_experiments(drop_prob: float, num_masks: int) -> dict[str, ExperimentSpec]:
    """Build E1..E9/I4 given a (p*, S*) chosen from the Pilot Study."""
    return {
        "E1_osfd_baseline": ExperimentSpec(
            "E1_osfd_baseline", AttackConfig(k=3.0, mask_enabled=False, augmentation="none")
        ),
        "E2_osfd_rrb": ExperimentSpec(
            "E2_osfd_rrb", AttackConfig(k=3.0, mask_enabled=False, augmentation="rrb")
        ),
        "E3_nrdm_control": ExperimentSpec(
            "E3_nrdm_control", AttackConfig(k=1.0, mask_enabled=False, augmentation="none")
        ),
        "E4_rapa_od_baseline": ExperimentSpec(
            "E4_rapa_od_baseline",
            AttackConfig(
                k=1.0,
                mask_enabled=True,
                drop_prob=drop_prob,
                num_masks=num_masks,
                augmentation="none",
            ),
        ),
        "E5_rapa_od_osfd_loss": ExperimentSpec(
            "E5_rapa_od_osfd_loss",
            AttackConfig(
                k=3.0,
                mask_enabled=True,
                drop_prob=drop_prob,
                num_masks=num_masks,
                augmentation="none",
            ),
        ),
        "I4_rapa_od_rrb": ExperimentSpec(
            "I4_rapa_od_rrb",
            AttackConfig(
                k=3.0,
                mask_enabled=True,
                drop_prob=drop_prob,
                num_masks=num_masks,
                augmentation="rrb",
            ),
        ),
        # Sec 5.2 augmentation comparison (new-plan.txt Phase 1), directly
        # comparable to E1 ("none") and E2 ("rrb") above: same loss (OSFD
        # k=3), no RaPA mask, same eps/alpha/iterations -- only the
        # augmentation axis varies.
        "E6_osfd_dim": ExperimentSpec(
            "E6_osfd_dim", AttackConfig(k=3.0, mask_enabled=False, augmentation="dim")
        ),
        "E7_osfd_ssa": ExperimentSpec(
            "E7_osfd_ssa",
            AttackConfig(k=3.0, mask_enabled=False, augmentation="ssa", num_masks=20),
        ),
        "E8_osfd_srs": ExperimentSpec(
            "E8_osfd_srs", AttackConfig(k=3.0, mask_enabled=False, augmentation="srs")
        ),
        # Option A: additive, not a replacement for RRB -- spectral band
        # attenuation on top of RRB left completely unmodified (theta=7,
        # blur included). Tests whether E8's gap to E2 on Group B/C comes
        # from SRS dropping RRB's blur/wider rotation rather than from the
        # spectral step itself.
        "E9_osfd_rrb_spectral": ExperimentSpec(
            "E9_osfd_rrb_spectral",
            AttackConfig(k=3.0, mask_enabled=False, augmentation="rrb_spectral"),
        ),
        # Phase 2 (idea.txt): relational contrast loss vs OSFD (E1), no RRB/
        # RaPA -- isolates whether attacking the object-vicinity relation
        # itself (Phase 1-validated in results/phase0_diagnostic.json)
        # transfers better than OSFD's absolute suppress/amplify. Stage
        # indices are the surrogate's own 4 backbone stages (0..3). Result:
        # NOGO -- REL_S23 (best of the 3) lost to E1 on 7/7 models, see
        # results/P2_rel_s23_go.json. Kept for the record / re-running.
        "REL_S3": ExperimentSpec(
            "REL_S3",
            AttackConfig(
                loss_type="rel", rel_stage_weights=(0.0, 0.0, 0.0, 1.0),
                mask_enabled=False, augmentation="none",
            ),
        ),
        "REL_S23": ExperimentSpec(
            "REL_S23",
            AttackConfig(
                loss_type="rel", rel_stage_weights=(0.0, 0.0, 0.5, 0.5),
                mask_enabled=False, augmentation="none",
            ),
        ),
        "REL_ALL": ExperimentSpec(
            "REL_ALL",
            AttackConfig(
                loss_type="rel", rel_stage_weights=(0.25, 0.25, 0.25, 0.25),
                mask_enabled=False, augmentation="none",
            ),
        ),
        # Phase 2 follow-up: does relational_diff (unbounded D, doesn't
        # saturate like standalone REL's bounded C) help as a regularizer
        # ON TOP of OSFD, rather than replacing it? Stage weights fixed to
        # S23 (best of the 3 standalone REL variants). lambda swept across
        # the calibrated ~50-85x scale gap between L_OSFD and L_relD.
        "HYBRID_L10": ExperimentSpec(
            "HYBRID_L10",
            AttackConfig(
                loss_type="osfd_rel_hybrid", k=3.0,
                rel_stage_weights=(0.0, 0.0, 0.5, 0.5), rel_lambda=10.0,
                mask_enabled=False, augmentation="none",
            ),
        ),
        "HYBRID_L30": ExperimentSpec(
            "HYBRID_L30",
            AttackConfig(
                loss_type="osfd_rel_hybrid", k=3.0,
                rel_stage_weights=(0.0, 0.0, 0.5, 0.5), rel_lambda=30.0,
                mask_enabled=False, augmentation="none",
            ),
        ),
        "HYBRID_L100": ExperimentSpec(
            "HYBRID_L100",
            AttackConfig(
                loss_type="osfd_rel_hybrid", k=3.0,
                rel_stage_weights=(0.0, 0.0, 0.5, 0.5), rel_lambda=100.0,
                mask_enabled=False, augmentation="none",
            ),
        ),
        # Phase 1b/2-followup: dense per-pixel object-to-context feature
        # misalignment, standalone (not a hybrid -- Phase 2's hybrid attempt
        # showed adding a second gradient direction to OSFD hurts rather than
        # helps, see HYBRID_* above). Phase 1b (results/P1b_prototype_diagnostic.json)
        # showed this quantity is architecture-invariant even where the
        # Phase 1 mean-pooled S_O/S_V metric had reversed (YOLOX stage 2), so
        # sweeping the same S3/S23/ALL stage-weight variants as standalone REL.
        "SPATIAL_S3": ExperimentSpec(
            "SPATIAL_S3",
            AttackConfig(
                loss_type="spatial", spatial_stage_weights=(0.0, 0.0, 0.0, 1.0),
                mask_enabled=False, augmentation="none",
            ),
        ),
        "SPATIAL_S23": ExperimentSpec(
            "SPATIAL_S23",
            AttackConfig(
                loss_type="spatial", spatial_stage_weights=(0.0, 0.0, 0.5, 0.5),
                mask_enabled=False, augmentation="none",
            ),
        ),
        "SPATIAL_ALL": ExperimentSpec(
            "SPATIAL_ALL",
            AttackConfig(
                loss_type="spatial", spatial_stage_weights=(0.25, 0.25, 0.25, 0.25),
                mask_enabled=False, augmentation="none",
            ),
        ),
        # Phase G0 (plan.md "RRB Gradient Mechanism Analysis"): does RRB's
        # E1->E2 transfer jump come from a stable cross-view gradient
        # subspace? Same K=5 RRB views MI-FGSM would sample per iteration
        # anyway (num_masks=5, matched compute budget across all four
        # variants) -- only the combine rule differs. CONS_SHUFFLE is the
        # spatial-shuffle control for CONS: same consensus-weight
        # distribution, positions permuted, isolating whether *where*
        # consensus occurs matters (not just how weighting affects gradient
        # magnitude). Run via scripts/run_g0_diagnostic.py, not run_attack.py
        # (needs per-image/per-checkpoint gradient logging run_attack.py
        # doesn't do).
        "RRB_K5_MEAN": ExperimentSpec(
            "RRB_K5_MEAN",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="rrb", num_masks=5, grad_combine="mean",
            ),
        ),
        "RRB_K5_CONS": ExperimentSpec(
            "RRB_K5_CONS",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="rrb", num_masks=5, grad_combine="consensus",
            ),
        ),
        "RRB_K5_DISAGREE": ExperimentSpec(
            "RRB_K5_DISAGREE",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="rrb", num_masks=5, grad_combine="disagree",
            ),
        ),
        "RRB_K5_CONS_SHUFFLE": ExperimentSpec(
            "RRB_K5_CONS_SHUFFLE",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="rrb", num_masks=5,
                grad_combine="consensus_shuffle",
            ),
        ),
        # Phase B1 (plan.md "RRB Component Ablation"): E1->E2 (adding RRB) is
        # the largest transferability jump in the project; this attributes it
        # to rotation, adaptive resize, additive noise (mislabeled "blur" in
        # the original OSFD code -- see rrb.py), or their interaction, by
        # running each component (and combinations) standalone against the
        # same E1/E2 reference points. Run via scripts/run_b1_ablation.py.
        "OSFD_ROT": ExperimentSpec(
            "OSFD_ROT", AttackConfig(k=3.0, mask_enabled=False, augmentation="rrb_rot")
        ),
        "OSFD_RESIZE": ExperimentSpec(
            "OSFD_RESIZE", AttackConfig(k=3.0, mask_enabled=False, augmentation="rrb_resize")
        ),
        "OSFD_NOISE": ExperimentSpec(
            "OSFD_NOISE", AttackConfig(k=3.0, mask_enabled=False, augmentation="rrb_noise")
        ),
        "OSFD_ROT_RESIZE": ExperimentSpec(
            "OSFD_ROT_RESIZE", AttackConfig(k=3.0, mask_enabled=False, augmentation="rrb_rot_resize")
        ),
        # Same config as E2 (full RRB), re-run at Phase B1's own pilot scale
        # so it's a fair same-scale reference point for the other 4 ablation
        # arms rather than reusing E2's 50-image/T=100 numbers directly.
        "OSFD_RRB_FULL": ExperimentSpec(
            "OSFD_RRB_FULL", AttackConfig(k=3.0, mask_enabled=False, augmentation="rrb")
        ),
        # Phase S1 (plan.md "Bidirectional/Shrink-aware RRB"): full RRB
        # (rotation+noise unchanged) with `random_occupancy_resize` replacing
        # `adaptive_random_resizing`, sampling occupancy ranges that actually
        # reach the occupancy~0.8 region Phase S0 found DINO-Swin-L benefits
        # from (RRB's own resize nets out to occupancy~[0.91,1.0], too
        # narrow to get there). RRB_ORIG reference = OSFD_RRB_FULL above
        # (same config, no separate entry needed).
        "RRB_SHRINK": ExperimentSpec(
            "RRB_SHRINK",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="rrb_occupancy", occ_low=0.7, occ_high=0.9
            ),
        ),
        "RRB_BIDIR": ExperimentSpec(
            "RRB_BIDIR",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="rrb_occupancy", occ_low=0.7, occ_high=1.0
            ),
        ),
        # Phase S2 (plan.md "Trajectory Consistency"): S1's RRB_SHRINK/BIDIR
        # both underperformed RRB_ORIG on every target including DINO-Swin-L
        # -- the opposite of what S0's fixed-scale diagnostic predicted.
        # Reconciles the two: resize-only (no rotation/noise, matching S0's
        # isolation), same occupancy range [0.7,0.9], only difference is
        # whether occupancy is drawn once per image (held for the whole
        # trajectory, like S0) or resampled every iteration (like S1/RRB).
        "FIXED_SHRINK": ExperimentSpec(
            "FIXED_SHRINK",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="fixed_shrink_per_image",
                occ_low=0.7, occ_high=0.9,
            ),
        ),
        "RANDOM_SHRINK": ExperimentSpec(
            "RANDOM_SHRINK",
            AttackConfig(
                k=3.0, mask_enabled=False, augmentation="random_shrink", occ_low=0.7, occ_high=0.9
            ),
        ),
        # Same computation as S0's scale=0.8 sweep point (fixed_scale=0.8 ->
        # occupancy=0.8, constant across every image) -- kept as a named
        # config for standalone reruns, but the S2 pilot script reuses S0's
        # already-computed results/S0_scale_transfer_sweep_v2.json instead.
        "FIXED_0.8": ExperimentSpec(
            "FIXED_0.8", AttackConfig(k=3.0, mask_enabled=False, augmentation="fixed_scale", fixed_scale=0.8)
        ),
    }
