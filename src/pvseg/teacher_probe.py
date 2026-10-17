"""Label-free diagnostics that separate global recognition from grid elicitation."""
from __future__ import annotations

import math
from typing import Any


DEFAULT_PRESENCE_PROMPT = """这是高分辨率遥感RGB影像。只判断整张图是否存在至少一处可辨识的太阳能光伏板，不要定位网格。
深色屋顶、水面、阴影和温室可能是干扰；证据不足时必须用unknown。
只输出一个JSON对象，不要输出Markdown或解释。对象必须恰好包含两个字段：
pv_probability（0到1之间的数值）和status（只能是"p"、"n"或"u"）。
pv_probability表示图中存在至少一处光伏板的可能性，不是回答自信度或面积占比。
"p"需有可辨识光伏证据且概率至少0.55；"n"表示明确未见光伏且概率至多0.18；否则用"u"。"""


def _protocol_valid(status: str, probability: float,
                    positive_probability: float, negative_probability: float) -> bool:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return False
    if status == "positive":
        return probability >= positive_probability
    if status == "negative":
        return probability <= negative_probability
    return status == "unknown"


def _grid_cells(view: dict[str, Any]) -> dict[tuple[int, int], tuple[str, float]]:
    cells = {}
    if view.get("status") != "ok":
        return cells
    for cell in view.get("grid", []) if isinstance(view.get("grid"), list) else []:
        try:
            row, col = int(cell["row"]), int(cell["col"])
            probability = float(cell.get("pv_probability", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= row < 4 or not 0 <= col < 4 or not math.isfinite(probability):
            continue
        if view.get("flipped"):
            col = 3 - col
        cells[(row, col)] = (str(cell.get("status", "unknown")).lower(), probability)
    return cells


def diagnose_grid_pair(views: dict[str, dict[str, Any]], positive_probability: float = .55,
                       negative_probability: float = .18) -> dict[str, Any]:
    """Summarize a global original/hflip pair in original-image coordinates."""
    aligned = {name: _grid_cells(views.get(name, {})) for name in ("original", "hflip")}
    per_view = {}
    for name, cells in aligned.items():
        counts = {status: sum(value[0] == status for value in cells.values())
                  for status in ("positive", "negative", "unknown")}
        violations = sum(
            not _protocol_valid(status, probability, positive_probability, negative_probability)
            for status, probability in cells.values()
        )
        per_view[name] = {
            "grid_cells": len(cells), "status_counts": counts,
            "complete": len(cells) == 16 and len(views.get(name, {}).get("grid", [])) == 16,
            "probability_status_violations": violations,
            "all_positive_anomaly": len(cells) == 16 and counts["positive"] == 16,
            "all_negative_anomaly": len(cells) == 16 and counts["negative"] == 16,
            "patch_count": views.get(name, {}).get("patch_count"),
            "elapsed_seconds": views.get(name, {}).get("elapsed_seconds"),
        }
    common = aligned["original"].keys() & aligned["hflip"].keys()
    exact = [key for key in common if aligned["original"][key][0] == aligned["hflip"][key][0]]
    conflicts = [key for key in common if {
        aligned["original"][key][0], aligned["hflip"][key][0]
    } == {"positive", "negative"}]
    positive_support, negative_support, protocol_valid_support = [], [], []
    for key in common:
        left, right = aligned["original"][key], aligned["hflip"][key]
        if left[0] == right[0] == "positive" and min(left[1], right[1]) >= positive_probability:
            positive_support.append(key)
        if left[0] == right[0] == "negative":
            # This mirrors the active fusion, where an explicit negative status
            # votes negative even when its generated number violates the prompt.
            negative_support.append(key)
        if (left[0] == right[0] and left[0] in {"positive", "negative"}
                and _protocol_valid(*left, positive_probability, negative_probability)
                and _protocol_valid(*right, positive_probability, negative_probability)):
            protocol_valid_support.append(key)
    return {
        "per_view": per_view,
        "complete_pair": all(row["complete"] for row in per_view.values()),
        "observed_pairs": len(common),
        "exact_status_agreement_cells": len(exact),
        "transform_status_agreement": len(exact) / max(len(common), 1),
        "positive_support_cells": len(positive_support),
        "negative_support_cells": len(negative_support),
        "pipeline_effective_support_coverage": (
            len(set(positive_support) | set(negative_support)) / 16
        ),
        "protocol_valid_support_coverage": len(protocol_valid_support) / 16,
        "conflicting_cells": [list(key) for key in sorted(conflicts)],
        "conflict_fraction": len(conflicts) / 16,
        "all_positive_views": sum(row["all_positive_anomaly"] for row in per_view.values()),
        "all_negative_views": sum(row["all_negative_anomaly"] for row in per_view.values()),
        "probability_status_violations": sum(
            row["probability_status_violations"] for row in per_view.values()
        ),
        "coverage_is_not_accuracy": True,
    }


def diagnose_presence_pair(views: dict[str, dict[str, Any]], positive_probability: float = .55,
                           negative_probability: float = .18) -> dict[str, Any]:
    decisions = []
    per_view = {}
    for name in ("original", "hflip"):
        view = views.get(name, {})
        status = str(view.get("status_label", "unknown"))
        probability = float(view.get("pv_probability", 0.0))
        valid = bool(view.get("parse_valid")) and _protocol_valid(
            status, probability, positive_probability, negative_probability
        )
        if valid:
            decisions.append(status)
        per_view[name] = {
            "status": status, "pv_probability": probability,
            "parse_valid": bool(view.get("parse_valid")), "protocol_valid": valid,
            "patch_count": view.get("patch_count"),
            "elapsed_seconds": view.get("elapsed_seconds"),
        }
    comparable = len(decisions) == 2
    transform_consistent = comparable and decisions[0] == decisions[1]
    return {
        "per_view": per_view,
        "parse_complete_pair": all(row["parse_valid"] for row in per_view.values()),
        "probability_status_violations": sum(
            row["parse_valid"] and not row["protocol_valid"] for row in per_view.values()),
        "comparable": comparable,
        "transform_consistent": transform_consistent,
        "usable_consistent_decision": transform_consistent and decisions[0] != "unknown",
        "conflict": comparable and {decisions[0], decisions[1]} == {"positive", "negative"},
        "self_probability_is_calibrated": False,
    }


def classify_probe(per_image: list[dict[str, Any]], max_grid_conflict: float) -> dict[str, str]:
    if not per_image or any(not row["presence"]["parse_complete_pair"] or not row["grid"]["complete_pair"]
                            for row in per_image):
        return {
            "diagnosis": "technical_or_output_schema_failure",
            "decision": "inspect_raw_responses_before_semantic_attribution",
            "reason": "Inference errors or incomplete/invalid responses cannot establish teacher recognition ability.",
        }
    if any(row["presence"]["probability_status_violations"] for row in per_image):
        return {
            "diagnosis": "generated_scores_violate_protocol",
            "decision": "do_not_use_generated_numbers_as_confidence",
            "reason": "Whole-image generated probabilities contradict their declared status semantics.",
        }
    presence_bad = any(not row["presence"]["transform_consistent"] for row in per_image)
    presence_abstained = any(not row["presence"]["usable_consistent_decision"] for row in per_image)
    grid_bad = any(row["grid"]["conflict_fraction"] > max_grid_conflict for row in per_image)
    schema_bad = any(row["grid"]["probability_status_violations"] for row in per_image)
    if presence_bad:
        return {
            "diagnosis": "global_teacher_or_transform_sensitivity",
            "decision": "stop_internvl_grid_teacher",
            "reason": "Whole-image PV decisions are not valid and flip-consistent under the same preprocessing.",
        }
    if presence_abstained:
        return {
            "diagnosis": "global_presence_abstained",
            "decision": "stop_internvl_grid_teacher",
            "reason": "Whole-image responses are flip-consistent but abstain, so they do not establish usable PV recognition.",
        }
    if grid_bad:
        return {
            "diagnosis": "grid_elicitation_instability",
            "decision": "replace_or_redesign_grid_teacher_before_pseudo_labels",
            "reason": "Whole-image decisions are stable but the 4x4 spatial response is not flip-consistent.",
        }
    if schema_bad:
        return {
            "diagnosis": "generated_scores_violate_protocol",
            "decision": "do_not_use_generated_numbers_as_confidence",
            "reason": "Statuses are usable for consistency checks, but generated probabilities violate their declared meaning.",
        }
    if any(row["grid"]["protocol_valid_support_coverage"] == 0 for row in per_image):
        return {
            "diagnosis": "no_usable_spatial_evidence",
            "decision": "stop_before_pseudo_labels",
            "reason": "Complete grids provide no agreed explicit positive or negative spatial evidence.",
        }
    for row in per_image:
        status = row["presence"]["per_view"]["original"]["status"]
        grid = row["grid"]
        if ((status == "positive" and grid["negative_support_cells"] == 16)
                or (status == "negative" and grid["positive_support_cells"] > 0)):
            return {
                "diagnosis": "presence_grid_disagreement",
                "decision": "redesign_spatial_teacher_before_pseudo_labels",
                "reason": "Stable whole-image and spatial decisions contradict each other; neither is ground truth.",
            }
    return {
        "diagnosis": "small_probe_passed",
        "decision": "eligible_for_bounded_local_view_and_pseudo_coverage_probe",
        "reason": "The probe checks stability and coverage only; it does not establish PV accuracy.",
    }
