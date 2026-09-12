from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AttackConfig:
    epsilon: float = 16 / 255
    spatial_step_size: float = 0.5 / 255
    reference_steps: int = 1000
    initialization: str = "zero"
    clusters: int = 10
    cluster_iterations: int = 100
    sinkhorn_iterations: int = 100
    sinkhorn_regularization: float = 0.1
    sinkhorn_tolerance: float = 1e-2
    detach_transport_plan: bool = True
    local_weight: float = 0.2
    text_weight: float = 1.0
    text_model_index: int = 1
    log_every: int = 50
    model_weight_update: str = "softmax_negative_preupdate_global_similarity"

    def validate(self) -> None:
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
            raise ValueError("attack.epsilon 必须位于 (0, 1]。")
        if not math.isfinite(self.spatial_step_size) or self.spatial_step_size <= 0:
            raise ValueError("attack.spatial_step_size 必须为正数。")
        if self.reference_steps < 0:
            raise ValueError("attack.reference_steps 不能为负数。")
        if self.initialization != "zero":
            raise ValueError("兼容实验只允许零初始化。")
        if self.clusters <= 0 or self.cluster_iterations <= 0:
            raise ValueError("聚类数和聚类迭代数必须为正整数。")
        if (
            self.sinkhorn_iterations <= 0
            or not math.isfinite(self.sinkhorn_regularization)
            or self.sinkhorn_regularization <= 0
        ):
            raise ValueError("最优传输迭代数和正则系数必须为正数。")
        if not math.isfinite(self.sinkhorn_tolerance) or self.sinkhorn_tolerance < 0:
            raise ValueError("最优传输停止容差不能为负数。")
        if (
            not math.isfinite(self.local_weight)
            or not math.isfinite(self.text_weight)
            or self.local_weight < 0
            or self.text_weight < 0
        ):
            raise ValueError("目标函数权重不能为负数。")
        if self.log_every <= 0:
            raise ValueError("attack.log_every 必须为正整数。")


@dataclass(frozen=True)
class SurrogateConfig:
    name: str
    pretrained: str
    hf_repo: str | None = None
    weight_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SurrogateConfig":
        return cls(
            name=str(raw["name"]),
            pretrained=str(raw.get("pretrained", "openai")),
            hf_repo=str(raw["hf_repo"]) if raw.get("hf_repo") else None,
            weight_path=str(raw["weight_path"]) if raw.get("weight_path") else None,
        )


@dataclass(frozen=True)
class SelectionConfig:
    mode: str = "strict_reference_compatibility"
    freeze_existing_ten: bool = True
    candidate_domains: tuple[str, ...] = ()
    retrieval_weight: float = 1.0
    contradiction_weight: float = 2.0
    correct_alignment_weight: float = 0.5
    lexical_penalty: float = 1_000_000.0
    on_topic_quantile_active: bool = False
    caption_source: str = "user_plus_assistant"
    preserve_duplicate_source_questions: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SelectionConfig":
        return cls(
            mode=str(raw.get("mode", "strict_reference_compatibility")),
            freeze_existing_ten=bool(raw.get("freeze_existing_ten", True)),
            candidate_domains=tuple(str(x) for x in raw.get("candidate_domains", ())),
            retrieval_weight=float(raw.get("retrieval_weight", 1.0)),
            contradiction_weight=float(raw.get("contradiction_weight", 2.0)),
            correct_alignment_weight=float(raw.get("correct_alignment_weight", 0.5)),
            lexical_penalty=float(raw.get("lexical_penalty", 1_000_000.0)),
            on_topic_quantile_active=bool(raw.get("on_topic_quantile_active", False)),
            caption_source=str(raw.get("caption_source", "user_plus_assistant")),
            preserve_duplicate_source_questions=bool(raw.get("preserve_duplicate_source_questions", True)),
        )

    def validate(self) -> None:
        if self.mode != "strict_reference_compatibility":
            raise ValueError("当前实现只支持严格参考兼容选图模式。")
        if self.on_topic_quantile_active:
            raise ValueError("参考选图没有启用主题分位数，兼容模式不得启用。")
        if self.caption_source != "user_plus_assistant":
            raise ValueError("兼容选图的候选描述必须由用户文本与助手文本拼接。")
        if self.lexical_penalty <= 0:
            raise ValueError("词汇门控惩罚必须为正数。")
        weights = (
            self.retrieval_weight,
            self.contradiction_weight,
            self.correct_alignment_weight,
            self.lexical_penalty,
        )
        if any(not math.isfinite(value) for value in weights):
            raise ValueError("选图权重必须是有限数值。")
        if not self.candidate_domains:
            raise ValueError("兼容选图至少需要一个候选领域。")


@dataclass(frozen=True)
class ProgressiveFrequencyConfig:
    enabled: bool = False
    time_fractions: tuple[float, ...] = (0.0, 0.3, 0.7)
    axis_ratios: tuple[float, ...] = (0.125, 0.25, 0.5)
    new_coefficient_initialization: str = "zero"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProgressiveFrequencyConfig":
        return cls(
            enabled=bool(raw.get("enabled", False)),
            time_fractions=tuple(float(x) for x in raw.get("time_fractions", (0.0, 0.3, 0.7))),
            axis_ratios=tuple(float(x) for x in raw.get("axis_ratios", (0.125, 0.25, 0.5))),
            new_coefficient_initialization=str(raw.get("new_coefficient_initialization", "zero")),
        )

    def validate(self) -> None:
        if len(self.time_fractions) != len(self.axis_ratios) or not self.axis_ratios:
            raise ValueError("渐进频带的时间节点与频率比例必须一一对应。")
        if self.time_fractions[0] != 0 or any(
            right <= left for left, right in zip(self.time_fractions, self.time_fractions[1:])
        ):
            raise ValueError("渐进频带时间节点必须从零开始并严格递增。")
        if any(not 0 < value <= 1 for value in self.time_fractions[1:]):
            raise ValueError("渐进频带时间节点必须位于 (0, 1]。")
        if any(not 0 < value <= 1 for value in self.axis_ratios):
            raise ValueError("频率方向比例必须位于 (0, 1]。")
        if any(right < left for left, right in zip(self.axis_ratios, self.axis_ratios[1:])):
            raise ValueError("渐进频带比例不能缩小。")
        if self.new_coefficient_initialization != "zero":
            raise ValueError("扩频时新增系数必须为零初始化。")


@dataclass(frozen=True)
class FrequencyConfig:
    transform: str = "orthonormal_dct_ii"
    parameterization: str = "compact_rgb_coefficients"
    height_width: str = "original_decoded_source"
    initial_axis_ratio: float = 0.25
    development_axis_ratios: tuple[float, ...] = (0.125, 0.25, 0.5)
    coefficient_direction: str = "sign_of_coefficient_gradient"
    spatial_direction_normalization: str = "linf"
    coefficient_feasibility: str = "spatial_clip_low_frequency_reproject_with_radial_fallback"
    maximum_reprojection_iterations: int = 1
    reprojection_tolerance: float = 1e-7
    image_feasibility: str = "clip_0_1"
    constant_component: bool = True
    final_saved_image_strictly_bandlimited: bool = False
    progressive: ProgressiveFrequencyConfig = field(default_factory=ProgressiveFrequencyConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrequencyConfig":
        return cls(
            transform=str(raw.get("transform", "orthonormal_dct_ii")),
            parameterization=str(raw.get("parameterization", "compact_rgb_coefficients")),
            height_width=str(raw.get("height_width", "original_decoded_source")),
            initial_axis_ratio=float(raw.get("initial_axis_ratio", 0.25)),
            development_axis_ratios=tuple(float(x) for x in raw.get("development_axis_ratios", (0.125, 0.25, 0.5))),
            coefficient_direction=str(raw.get("coefficient_direction", "sign_of_coefficient_gradient")),
            spatial_direction_normalization=str(raw.get("spatial_direction_normalization", "linf")),
            coefficient_feasibility=str(
                raw.get(
                    "coefficient_feasibility",
                    "spatial_clip_low_frequency_reproject_with_radial_fallback",
                )
            ),
            maximum_reprojection_iterations=raw.get("maximum_reprojection_iterations", 1),
            reprojection_tolerance=float(raw.get("reprojection_tolerance", 1e-7)),
            image_feasibility=str(raw.get("image_feasibility", "clip_0_1")),
            constant_component=bool(raw.get("constant_component", True)),
            final_saved_image_strictly_bandlimited=bool(raw.get("final_saved_image_strictly_bandlimited", False)),
            progressive=ProgressiveFrequencyConfig.from_mapping(raw.get("progressive", {})),
        )

    def validate(self) -> None:
        expected = {
            "transform": (self.transform, "orthonormal_dct_ii"),
            "parameterization": (self.parameterization, "compact_rgb_coefficients"),
            "height_width": (self.height_width, "original_decoded_source"),
            "coefficient_direction": (self.coefficient_direction, "sign_of_coefficient_gradient"),
            "spatial_direction_normalization": (self.spatial_direction_normalization, "linf"),
            "coefficient_feasibility": (
                self.coefficient_feasibility,
                "spatial_clip_low_frequency_reproject_with_radial_fallback",
            ),
            "image_feasibility": (self.image_feasibility, "clip_0_1"),
        }
        wrong = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if wrong:
            raise ValueError(f"当前实现不支持这些频域配置变体：{', '.join(wrong)}。")
        if not 0 < self.initial_axis_ratio <= 1:
            raise ValueError("frequency.initial_axis_ratio 必须位于 (0, 1]。")
        if any(not 0 < value <= 1 for value in self.development_axis_ratios):
            raise ValueError("开发频率比例必须位于 (0, 1]。")
        if (
            isinstance(self.maximum_reprojection_iterations, bool)
            or not isinstance(self.maximum_reprojection_iterations, int)
            or not 1 <= self.maximum_reprojection_iterations <= 16
        ):
            raise ValueError("frequency.maximum_reprojection_iterations 必须位于 [1, 16]。")
        if not math.isfinite(self.reprojection_tolerance) or self.reprojection_tolerance < 0:
            raise ValueError("frequency.reprojection_tolerance 必须为有限非负数。")
        if self.final_saved_image_strictly_bandlimited:
            raise ValueError("范围裁剪和整数量化后不能保证落盘图片严格限频。")
        self.progressive.validate()


@dataclass(frozen=True)
class SharedEngineeringConfig:
    cache_target_features: bool = True
    cache_target_cluster_centers: bool = True
    load_surrogates_once: bool = True
    sequential_surrogate_gradients: bool = True
    separate_selection_generation_evaluation: bool = True
    mixed_precision_enabled: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SharedEngineeringConfig":
        values = {name: bool(raw.get(name, default)) for name, default in {
            "cache_target_features": True,
            "cache_target_cluster_centers": True,
            "load_surrogates_once": True,
            "sequential_surrogate_gradients": True,
            "separate_selection_generation_evaluation": True,
            "mixed_precision_enabled": False,
        }.items()}
        return cls(**values)


@dataclass(frozen=True)
class EvaluationConfig:
    primary_encoder: str = "CLIP-B-32"
    primary_model_id: str = "openai/clip-vit-base-patch32"
    embedding_dimension: int = 512
    precision_cuda: str = "fp16"
    precision_cpu: str = "fp32"
    read_saved_png: bool = True
    use_primary_evaluator_gradient: bool = False
    additional_encoders_enabled: bool = False
    optional_existing_encoders: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvaluationConfig":
        return cls(
            primary_encoder=str(raw.get("primary_encoder", "CLIP-B-32")),
            primary_model_id=str(raw.get("primary_model_id", "openai/clip-vit-base-patch32")),
            embedding_dimension=int(raw.get("embedding_dimension", 512)),
            precision_cuda=str(raw.get("precision_cuda", "fp16")),
            precision_cpu=str(raw.get("precision_cpu", "fp32")),
            read_saved_png=bool(raw.get("read_saved_png", True)),
            use_primary_evaluator_gradient=bool(raw.get("use_primary_evaluator_gradient", False)),
            additional_encoders_enabled=bool(raw.get("additional_encoders_enabled", False)),
            optional_existing_encoders=tuple(str(x) for x in raw.get("optional_existing_encoders", ())),
        )

    def validate(self) -> None:
        if not self.read_saved_png:
            raise ValueError("正式评估必须重新读取已保存图片。")
        if self.use_primary_evaluator_gradient:
            raise ValueError("主评估器不得参与攻击反向传播。")
        if self.primary_encoder != "CLIP-B-32":
            raise ValueError("首轮兼容评估器必须是 CLIP-B-32。")
        if self.primary_model_id != "openai/clip-vit-base-patch32":
            raise ValueError("首轮兼容评估模型标识不一致。")
        if self.embedding_dimension != 512:
            raise ValueError("首轮主评估向量维度必须为 512。")
        if self.precision_cuda != "fp16" or self.precision_cpu != "fp32":
            raise ValueError("首轮主评估精度必须是图形处理器 fp16、处理器 fp32。")
        if self.additional_encoders_enabled:
            raise ValueError("首轮兼容实验不得默认启用额外评估器。")


@dataclass(frozen=True)
class ExperimentConfig:
    smoke_steps: int = 25
    development_unique_sources: int = 2
    development_max_steps: int = 200
    compatibility_rows: int = 10
    optional_validation_max_unique_sources: int = 20
    timing_repetitions: int = 3
    equal_time_reference: str = "median_fair_pixel_1000_step_loop_per_sample"
    checkpoints_budget_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    report_preparation_and_total_attack: bool = True
    report_end_to_end: bool = True
    online_evaluator_stopping_enabled: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            smoke_steps=int(raw.get("smoke_steps", 25)),
            development_unique_sources=int(raw.get("development_unique_sources", 2)),
            development_max_steps=int(raw.get("development_max_steps", 200)),
            compatibility_rows=int(raw.get("compatibility_rows", 10)),
            optional_validation_max_unique_sources=int(raw.get("optional_validation_max_unique_sources", 20)),
            timing_repetitions=int(raw.get("timing_repetitions", 3)),
            equal_time_reference=str(raw.get("equal_time_reference", "median_fair_pixel_1000_step_loop_per_sample")),
            checkpoints_budget_fractions=tuple(float(x) for x in raw.get("checkpoints_budget_fractions", (0.25, 0.5, 0.75, 1.0))),
            report_preparation_and_total_attack=bool(raw.get("report_preparation_and_total_attack", True)),
            report_end_to_end=bool(raw.get("report_end_to_end", True)),
            online_evaluator_stopping_enabled=bool(raw.get("online_evaluator_stopping_enabled", False)),
        )

    def validate(self) -> None:
        if any(not 0 < value <= 1 for value in self.checkpoints_budget_fractions):
            raise ValueError("检查点预算比例必须位于 (0, 1]。")
        if tuple(sorted(set(self.checkpoints_budget_fractions))) != self.checkpoints_budget_fractions:
            raise ValueError("检查点预算比例必须严格递增且不能重复。")
        if self.online_evaluator_stopping_enabled:
            raise ValueError("首轮协议不允许在线查询主评估器停止。")
        positive_counts = (
            self.smoke_steps,
            self.development_unique_sources,
            self.development_max_steps,
            self.compatibility_rows,
            self.optional_validation_max_unique_sources,
            self.timing_repetitions,
        )
        if any(value <= 0 for value in positive_counts):
            raise ValueError("实验步数、样本数和计时重复数必须为正整数。")
        if self.equal_time_reference != "median_fair_pixel_1000_step_loop_per_sample":
            raise ValueError("首轮同时间预算必须来自公平像素逐样本循环时间中位数。")


@dataclass(frozen=True)
class AcceptanceConfig:
    same_time_mean_target_gain_advantage: float = 0.01
    same_time_total_attack_ratio_max: float = 1.05
    same_time_peak_memory_ratio_max: float = 1.05
    equivalent_effect_cosine_tolerance: float = 0.005
    equivalent_effect_reach_rate_min: float = 0.9
    equivalent_effect_median_total_attack_time_ratio_max: float = 0.8
    mean_query_cosine_drop_max: float = 0.01
    mean_source_adversarial_cosine_drop_max: float = 0.02
    decoded_max_pixel_difference: int = 16
    exploratory_memory_reduction_fraction: float = 0.2

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AcceptanceConfig":
        defaults = cls()
        values: dict[str, Any] = {}
        for name in defaults.__dataclass_fields__:
            value = raw.get(name, getattr(defaults, name))
            values[name] = int(value) if name == "decoded_max_pixel_difference" else float(value)
        return cls(**values)

    def validate(self) -> None:
        numeric = (
            self.same_time_mean_target_gain_advantage,
            self.same_time_total_attack_ratio_max,
            self.same_time_peak_memory_ratio_max,
            self.equivalent_effect_cosine_tolerance,
            self.equivalent_effect_reach_rate_min,
            self.equivalent_effect_median_total_attack_time_ratio_max,
            self.mean_query_cosine_drop_max,
            self.mean_source_adversarial_cosine_drop_max,
            self.exploratory_memory_reduction_fraction,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("验收阈值必须是有限数值。")
        if self.same_time_mean_target_gain_advantage < 0:
            raise ValueError("同时间目标增益优势阈值不能为负数。")
        if self.same_time_total_attack_ratio_max <= 0 or self.same_time_peak_memory_ratio_max <= 0:
            raise ValueError("同时间成本比例阈值必须为正数。")
        if self.equivalent_effect_cosine_tolerance < 0:
            raise ValueError("等效效果余弦容差不能为负数。")
        if not 0 <= self.equivalent_effect_reach_rate_min <= 1:
            raise ValueError("等效效果达标率必须位于 [0, 1]。")
        if self.equivalent_effect_median_total_attack_time_ratio_max <= 0:
            raise ValueError("等效效果时间比例必须为正数。")
        if self.mean_query_cosine_drop_max < 0 or self.mean_source_adversarial_cosine_drop_max < 0:
            raise ValueError("质量保护允许下降值不能为负数。")
        if not 0 <= self.decoded_max_pixel_difference <= 255:
            raise ValueError("落盘像素差上限必须位于 [0, 255]。")
        if not 0 <= self.exploratory_memory_reduction_fraction <= 1:
            raise ValueError("探索显存下降比例必须位于 [0, 1]。")


@dataclass(frozen=True)
class RuntimeConfig:
    allow_downloads: bool = False
    allow_device_fallback: bool = False
    require_true_local_tokens: bool = True
    hash_model_weights: bool = True
    cache_directory: str = "outputs/cache"
    device_memory_sample_interval_seconds: float = 0.05

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RuntimeConfig":
        return cls(
            allow_downloads=bool(raw.get("allow_downloads", False)),
            allow_device_fallback=bool(raw.get("allow_device_fallback", False)),
            require_true_local_tokens=bool(raw.get("require_true_local_tokens", True)),
            hash_model_weights=bool(raw.get("hash_model_weights", True)),
            cache_directory=str(raw.get("cache_directory", "outputs/cache")),
            device_memory_sample_interval_seconds=float(raw.get("device_memory_sample_interval_seconds", 0.05)),
        )

    def validate(self) -> None:
        if (
            not math.isfinite(self.device_memory_sample_interval_seconds)
            or self.device_memory_sample_interval_seconds <= 0
        ):
            raise ValueError("设备显存采样间隔必须为有限正数。")
        if not self.cache_directory.strip():
            raise ValueError("运行缓存目录不能为空。")


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    status: str
    reference_root: str
    reference_manifest: str
    seed: int
    device: str
    attack_precision: str
    memgallery_root: str
    dataset: str
    selection: SelectionConfig
    attack: AttackConfig
    surrogates: tuple[SurrogateConfig, ...]
    frequency: FrequencyConfig
    shared_engineering: SharedEngineeringConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig
    acceptance: AcceptanceConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    pending_verification: tuple[str, ...] = ()
    config_path: Path | None = field(default=None, compare=False, repr=False)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"不支持配置版本 {self.schema_version}。")
        if self.status not in {"implemented_not_benchmarked", "implemented_and_benchmarked"}:
            raise ValueError("请使用可执行配置；设计草案配置不能直接运行攻击。")
        if self.device not in {"cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("device 只支持 cpu、cuda 或 cuda:序号。")
        if self.attack_precision != "fp32":
            raise ValueError("首轮攻击必须使用 fp32。")
        if len(self.surrogates) != 3:
            raise ValueError("主实验必须恰好使用三个代理模型。")
        if not 0 <= self.attack.text_model_index < len(self.surrogates):
            raise ValueError("文本代理索引越界。")
        self.selection.validate()
        self.attack.validate()
        self.frequency.validate()
        self.evaluation.validate()
        self.experiment.validate()
        self.acceptance.validate()
        self.runtime.validate()
        if self.runtime.allow_device_fallback:
            raise ValueError("正式配置不得在设备不可用时静默回退。")
        if self.shared_engineering.mixed_precision_enabled:
            raise ValueError("首轮公平对照不启用混合精度。")
        required_shared = {
            "cache_target_features": self.shared_engineering.cache_target_features,
            "cache_target_cluster_centers": self.shared_engineering.cache_target_cluster_centers,
            "load_surrogates_once": self.shared_engineering.load_surrogates_once,
            "sequential_surrogate_gradients": self.shared_engineering.sequential_surrogate_gradients,
            "separate_selection_generation_evaluation": self.shared_engineering.separate_selection_generation_evaluation,
        }
        disabled = [name for name, enabled in required_shared.items() if not enabled]
        if disabled:
            raise ValueError(f"公平主配置缺少公共工程条件：{', '.join(disabled)}。")

    def resolve_project_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        base = self.config_path.parent.parent if self.config_path else Path.cwd()
        return (base / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self, exclude={"config_path"})


def _to_plain(value: Any, *, exclude: set[str] | None = None) -> Any:
    exclude = exclude or set()
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _to_plain(getattr(value, name), exclude=exclude)
            for name in value.__dataclass_fields__
            if name not in exclude
        }
    if isinstance(value, (tuple, list)):
        return [_to_plain(item, exclude=exclude) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    project_root = Path(__file__).resolve().parents[2]
    project_candidate = project_root / candidate
    if project_candidate.is_file():
        return project_candidate.resolve()
    raise FileNotFoundError(f"配置文件不存在：{candidate}")


def load_config(path: str | Path = "configs/default.json", *, validate: bool = True) -> ProjectConfig:
    resolved = resolve_config_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("配置根节点必须是对象。")

    cfg = ProjectConfig(
        schema_version=int(raw.get("schema_version", 1)),
        status=str(raw.get("status", "")),
        reference_root=str(raw.get("reference_root", "")),
        reference_manifest=str(raw.get("reference_manifest", "manifests/reference_snapshot.json")),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cuda")),
        attack_precision=str(raw.get("attack_precision", "fp32")),
        memgallery_root=str(raw.get("memgallery_root", "")),
        dataset=str(raw.get("dataset", "AI_Robotics_Automation_Future_Tech")),
        selection=SelectionConfig.from_mapping(raw.get("selection", {})),
        attack=AttackConfig(**raw.get("attack", {})),
        surrogates=tuple(SurrogateConfig.from_mapping(item) for item in raw.get("surrogates", ())),
        frequency=FrequencyConfig.from_mapping(raw.get("frequency", {})),
        shared_engineering=SharedEngineeringConfig.from_mapping(raw.get("shared_engineering", {})),
        evaluation=EvaluationConfig.from_mapping(raw.get("evaluation", {})),
        experiment=ExperimentConfig.from_mapping(raw.get("experiment", {})),
        acceptance=AcceptanceConfig.from_mapping(raw.get("acceptance", {})),
        runtime=RuntimeConfig.from_mapping(raw.get("runtime", {})),
        pending_verification=tuple(str(x) for x in raw.get("pending_verification", ())),
        config_path=resolved,
    )
    if validate:
        cfg.validate()
    return cfg


def select_items(values: Sequence[Any], indices: Sequence[int] | None) -> list[Any]:
    if indices is None:
        return list(values)
    return [values[index] for index in indices]
