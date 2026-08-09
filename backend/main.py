from __future__ import annotations

import io
import math
import time
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from backend.decision_engine import determine_solder_decision


# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_yolo11s.pt"
OVERLAY_DIR = PROJECT_ROOT / "outputs" / "overlays"

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 대상 납땜 Pair 선택 설정
# ---------------------------------------------------------
# 기존 고정 ROI 좌표 대신, 모델이 이미지 전체에서 검출한 납땜 객체 중
# "하나의 R0805 부품을 이루는 좌/우 납땜 Pair"를 동적으로 선택한다.
#
# 핵심 원칙
# 1) 좌/우 객체가 수평 방향으로 충분히 떨어져 있어야 한다.
# 2) 두 객체의 높이(Y 중심)는 비슷해야 한다.
# 3) 너무 멀리 떨어진 객체끼리는 같은 부품 Pair로 보지 않는다.
# 4) 여러 Pair 후보가 있으면 Pair의 중간점이 이미지 중앙에 가깝고,
#    수평 정렬이 좋으며, 두 검출의 confidence가 높은 후보를 우선한다.
#
# 아래 값은 특정 사진의 픽셀 좌표가 아니라 Pair의 기하학적 관계를
# 정의하는 일반화된 후처리 기준이다.
PAIR_CANDIDATE_MIN_CONFIDENCE = 0.05
PAIR_MIN_HORIZONTAL_IMAGE_RATIO = 0.04
PAIR_MAX_HORIZONTAL_IMAGE_RATIO = 0.58
PAIR_MAX_VERTICAL_IMAGE_RATIO = 0.14
PAIR_MIN_HORIZONTAL_TO_VERTICAL = 1.25
PAIR_MIN_SEPARATION_TO_MEAN_BOX_WIDTH = 0.75


# ---------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
    )

try:
    model = YOLO(str(MODEL_PATH))
except Exception as exc:
    raise RuntimeError(
        f"YOLO 모델을 불러오지 못했습니다: {exc}"
    ) from exc


# ---------------------------------------------------------
# FastAPI 앱 생성
# ---------------------------------------------------------

app = FastAPI(
    title="LineGate AOI Solder Inspection API",
    description=(
        "V2 및 V2.1 이미지를 입력받아 납땜 상태를 "
        "Segmentation 모델로 분석하는 PoC API"
    ),
    version="0.2.0",
)


# ---------------------------------------------------------
# 기본 엔드포인트
# ---------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "LineGate AOI",
        "message": "Solder Inspection API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """서버와 모델 상태를 확인한다."""

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_name": MODEL_PATH.name,
        "model_task": model.task,
        "classes": model.names,
        "target_filter": "dynamic_solder_pair",
    }


# ---------------------------------------------------------
# 추론 엔드포인트
# ---------------------------------------------------------

@app.post("/predict/solder")
async def predict_solder(
    file: UploadFile = File(...),
    view: str = Form(...),
    confidence_threshold: float = Form(0.70),
) -> dict[str, Any]:
    """
    V2 또는 V2.1 이미지를 분석한다.

    입력:
    - file: jpg, jpeg, png 이미지
    - view: V2 또는 V2.1
    - confidence_threshold: 운영 판정 기준

    출력:
    - 전체 모델 검출 결과
    - 동적으로 선택된 대상 R0805 납땜 Pair
    - confidence / bbox / polygon / mask 면적
    - RELEASE/HOLD/REVIEW
    - 선택된 Pair만 표시한 Overlay 이미지 URL
    """

    normalized_view = normalize_view(view)

    if not 0.0 < confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="confidence_threshold는 0보다 크고 1 이하여야 합니다.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="파일 이름이 없습니다.",
        )

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="jpg, jpeg, png 이미지만 업로드할 수 있습니다.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="업로드된 파일이 비어 있습니다.",
        )

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="유효한 이미지 파일이 아닙니다.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"이미지를 읽는 중 오류가 발생했습니다: {exc}",
        ) from exc

    image_rgb = np.array(pil_image)

    request_id = f"REQ-{uuid.uuid4().hex[:12].upper()}"

    inference_started_at = time.perf_counter()

    try:
        results = model.predict(
            source=image_rgb,
            # 운영 기준보다 낮은 후보도 먼저 받아야
            # REVIEW 및 Pair 선택 후처리를 적용할 수 있다.
            conf=0.01,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"모델 추론 중 오류가 발생했습니다: {exc}",
        ) from exc

    inference_time_ms = (
        time.perf_counter() - inference_started_at
    ) * 1000

    if not results:
        raise HTTPException(
            status_code=500,
            detail="모델이 추론 결과를 반환하지 않았습니다.",
        )

    result = results[0]

    # 1) 모델이 이미지 전체에서 찾은 원본 검출 결과
    raw_detections = extract_detections(result)

    # 2) 특정 좌표 ROI를 쓰지 않고, 좌/우 납땜의 공간 관계를 이용해
    #    이번 검사 대상 R0805에 해당하는 한 Pair를 동적으로 선택한다.
    (
        detections,
        filtered_out_detections,
        target_selection,
    ) = select_target_solder_pair(
        detections=raw_detections,
        image_width=pil_image.width,
        image_height=pil_image.height,
    )

    # 3) 선택된 대상 Pair만 RELEASE / HOLD / REVIEW 판정에 반영한다.
    #    Pair를 만들지 못하면 detections=[] 이므로 보수적으로 REVIEW가 된다.
    decision = determine_solder_decision(
        detections=detections,
        confidence_threshold=confidence_threshold,
    )

    overlay_filename = f"{request_id}.jpg"
    overlay_path = OVERLAY_DIR / overlay_filename

    # 주변 부품 검출은 화면에서도 제외하고 선택된 대상 Pair만 그린다.
    save_target_pair_overlay(
        image_rgb=image_rgb,
        detections=detections,
        target_selection=target_selection,
        output_path=overlay_path,
    )

    top_detection = (
        max(
            detections,
            key=lambda item: item["confidence"],
        )
        if detections
        else None
    )

    confident_detections = [
        detection
        for detection in detections
        if detection["confidence"] >= confidence_threshold
    ]

    defect_classes = {
        "exc_solder",
        "poor_solder",
        "spike",
    }

    has_defect = any(
        detection["class_name"] in defect_classes
        for detection in confident_detections
    )

    return {
        "request_id": request_id,
        "filename": file.filename,
        "view": normalized_view,
        "model": {
            "name": MODEL_PATH.name,
            "task": model.task,
            "classes": model.names,
        },
        "image": {
            "width": pil_image.width,
            "height": pil_image.height,
        },
        # 기존 프론트엔드 호환을 위해 roi 키는 유지하되,
        # 고정 좌표 ROI가 사용되지 않는다는 점을 명시한다.
        "roi": {
            "mode": "disabled",
            "rule": (
                "고정 좌표 ROI 대신 좌/우 납땜의 공간 관계를 이용한 "
                "dynamic target-pair selection을 사용"
            ),
        },
        "target_selection": target_selection,
        "inference": {
            "time_ms": round(inference_time_ms, 2),
            "confidence_threshold": confidence_threshold,
            "pair_candidate_min_confidence": (
                PAIR_CANDIDATE_MIN_CONFIDENCE
            ),
        },
        "detections": detections,
        "filtered_out_detections": filtered_out_detections,
        "summary": {
            "raw_detection_count": len(raw_detections),
            "total_detection_count": len(detections),
            "filtered_out_detection_count": len(filtered_out_detections),
            "confident_detection_count": len(confident_detections),
            "target_pair_found": bool(detections),
            "top_class": (
                top_detection["class_name"]
                if top_detection
                else None
            ),
            "top_confidence": (
                top_detection["confidence"]
                if top_detection
                else None
            ),
            "has_defect": has_defect,
        },
        "decision": decision,
        "overlay": {
            "filename": overlay_filename,
            "url": f"/overlays/{overlay_filename}",
        },
        "notice": (
            "본 결과는 PoC 의사결정 지원 결과이며, "
            "공정 원인이나 실제 손실을 확정하지 않습니다. "
            "이미지 전체 모델 검출 중 대상 R0805의 좌/우 납땜 Pair로 "
            "선택된 객체만 최종 판정 및 Overlay에 사용됩니다."
        ),
    }


# ---------------------------------------------------------
# Overlay 조회
# ---------------------------------------------------------

@app.get("/overlays/{filename}")
def get_overlay(filename: str) -> FileResponse:
    """저장된 추론 Overlay 이미지를 반환한다."""

    safe_filename = Path(filename).name
    overlay_path = OVERLAY_DIR / safe_filename

    if not overlay_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Overlay 이미지를 찾을 수 없습니다.",
        )

    return FileResponse(
        path=overlay_path,
        media_type="image/jpeg",
        filename=safe_filename,
    )


# ---------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------

def normalize_view(view: str) -> str:
    """V21 등의 입력을 V2.1로 통일한다."""

    normalized = view.strip().upper().replace("_", "").replace("-", "")

    mapping = {
        "V2": "V2",
        "V21": "V2.1",
        "V2.1": "V2.1",
    }

    if normalized not in mapping:
        raise HTTPException(
            status_code=400,
            detail="view는 V2 또는 V2.1이어야 합니다.",
        )

    return mapping[normalized]


def extract_detections(result: Any) -> list[dict[str, Any]]:
    """Ultralytics 결과에서 웹에 필요한 정보를 추출한다."""

    detections: list[dict[str, Any]] = []

    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        return detections

    for index in range(len(boxes)):
        box = boxes[index]

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        bbox_values = box.xyxy[0].detach().cpu().tolist()
        bbox = [
            round(float(value), 2)
            for value in bbox_values
        ]

        polygon: list[list[float]] = []
        mask_area_px: int | None = None

        if masks is not None and index < len(masks.xy):
            polygon_array = masks.xy[index]

            polygon = [
                [
                    round(float(point[0]), 2),
                    round(float(point[1]), 2),
                ]
                for point in polygon_array
            ]

            if index < len(masks.data):
                mask_data = (
                    masks.data[index]
                    .detach()
                    .cpu()
                    .numpy()
                )

                mask_area_px = int(
                    np.count_nonzero(mask_data > 0.5)
                )

        class_name = str(result.names[class_id])

        detections.append(
            {
                "detection_id": index + 1,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": bbox,
                "polygon": polygon,
                "mask_area_px": mask_area_px,
            }
        )

    return detections


def get_bbox_geometry(
    detection: dict[str, Any],
) -> dict[str, float] | None:
    """Pair 선택에 사용할 안정적인 bbox 기하 정보를 계산한다."""

    bbox = detection.get("bbox_xyxy") or []

    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = map(float, bbox)

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": (x1 + x2) / 2.0,
        "center_y": (y1 + y2) / 2.0,
        "width": width,
        "height": height,
        "area": width * height,
    }


def build_pair_candidate(
    left_detection: dict[str, Any],
    right_detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """
    두 detection이 같은 R0805의 좌/우 납땜 Pair로 볼 수 있는지 검사하고
    가능하면 낮을수록 좋은 pair score를 반환한다.

    특정 이미지의 절대 좌표는 사용하지 않고, 두 검출 사이의 상대적
    수평 거리·수직 정렬·크기·confidence와 Pair 중심 위치를 사용한다.
    """

    left_geometry = get_bbox_geometry(left_detection)
    right_geometry = get_bbox_geometry(right_detection)

    if left_geometry is None or right_geometry is None:
        return None

    # 실제 좌/우 순서를 보장한다.
    if left_geometry["center_x"] > right_geometry["center_x"]:
        left_detection, right_detection = right_detection, left_detection
        left_geometry, right_geometry = right_geometry, left_geometry

    horizontal_distance = (
        right_geometry["center_x"] - left_geometry["center_x"]
    )
    vertical_distance = abs(
        right_geometry["center_y"] - left_geometry["center_y"]
    )

    mean_box_width = (
        left_geometry["width"] + right_geometry["width"]
    ) / 2.0

    min_horizontal_distance = max(
        image_width * PAIR_MIN_HORIZONTAL_IMAGE_RATIO,
        mean_box_width * PAIR_MIN_SEPARATION_TO_MEAN_BOX_WIDTH,
    )
    max_horizontal_distance = (
        image_width * PAIR_MAX_HORIZONTAL_IMAGE_RATIO
    )
    max_vertical_distance = max(
        image_height * PAIR_MAX_VERTICAL_IMAGE_RATIO,
        (
            left_geometry["height"] + right_geometry["height"]
        ) / 2.0,
    )

    if horizontal_distance < min_horizontal_distance:
        return None

    if horizontal_distance > max_horizontal_distance:
        return None

    if vertical_distance > max_vertical_distance:
        return None

    if horizontal_distance < (
        vertical_distance * PAIR_MIN_HORIZONTAL_TO_VERTICAL
    ):
        return None

    pair_center_x = (
        left_geometry["center_x"] + right_geometry["center_x"]
    ) / 2.0
    pair_center_y = (
        left_geometry["center_y"] + right_geometry["center_y"]
    ) / 2.0

    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0

    # 이미지 크기에 독립적인 중심 거리.
    normalized_center_distance = math.sqrt(
        (
            (pair_center_x - image_center_x)
            / max(image_width / 2.0, 1.0)
        ) ** 2
        + (
            (pair_center_y - image_center_y)
            / max(image_height / 2.0, 1.0)
        ) ** 2
    )

    normalized_alignment = (
        vertical_distance / max(max_vertical_distance, 1.0)
    )

    # 같은 부품의 좌/우 접합부는 너무 멀리 떨어진 cross-component Pair보다
    # 상대적으로 compact하다는 점을 약한 점수로 반영한다.
    normalized_compactness = (
        horizontal_distance / max(float(image_width), 1.0)
    )

    left_area = left_geometry["area"]
    right_area = right_geometry["area"]
    size_imbalance = min(
        1.0,
        abs(math.log((left_area + 1.0) / (right_area + 1.0))) / 2.5,
    )

    mean_confidence = (
        float(left_detection.get("confidence", 0.0))
        + float(right_detection.get("confidence", 0.0))
    ) / 2.0
    confidence_penalty = 1.0 - mean_confidence

    # 낮을수록 좋은 점수.
    # 중앙성은 촬영 시 대상 R0805를 주요 프레임에 두는 데이터 특성을 이용하되,
    # 고정 ROI처럼 특정 좌표에 결과를 강제로 묶지는 않는다.
    score = (
        0.52 * normalized_center_distance
        + 0.20 * normalized_alignment
        + 0.13 * normalized_compactness
        + 0.05 * size_imbalance
        + 0.10 * confidence_penalty
    )

    return {
        "left_detection": left_detection,
        "right_detection": right_detection,
        "left_geometry": left_geometry,
        "right_geometry": right_geometry,
        "pair_center_xy": [
            round(pair_center_x, 2),
            round(pair_center_y, 2),
        ],
        "horizontal_distance_px": round(horizontal_distance, 2),
        "vertical_distance_px": round(vertical_distance, 2),
        "mean_confidence": round(mean_confidence, 4),
        "score": round(score, 6),
    }


def bbox_iou(
    first_geometry: dict[str, float],
    second_geometry: dict[str, float],
) -> float:
    """두 bbox의 IoU를 계산한다."""

    intersection_x1 = max(first_geometry["x1"], second_geometry["x1"])
    intersection_y1 = max(first_geometry["y1"], second_geometry["y1"])
    intersection_x2 = min(first_geometry["x2"], second_geometry["x2"])
    intersection_y2 = min(first_geometry["y2"], second_geometry["y2"])

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    union_area = (
        first_geometry["area"]
        + second_geometry["area"]
        - intersection_area
    )

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def belongs_to_anchor_cluster(
    detection: dict[str, Any],
    anchor_detection: dict[str, Any],
) -> bool:
    """
    detection이 anchor와 같은 납땜 접합부를 설명하는 중복/보조 검출인지
    판단한다.

    같은 접합부에서 서로 다른 클래스나 mask가 겹쳐 나오는 경우를
    보존하기 위해 bbox IoU와 중심 거리 두 조건을 함께 사용한다.
    """

    geometry = get_bbox_geometry(detection)
    anchor_geometry = get_bbox_geometry(anchor_detection)

    if geometry is None or anchor_geometry is None:
        return False

    overlap = bbox_iou(geometry, anchor_geometry)

    center_distance = math.hypot(
        geometry["center_x"] - anchor_geometry["center_x"],
        geometry["center_y"] - anchor_geometry["center_y"],
    )

    anchor_diagonal = math.hypot(
        anchor_geometry["width"],
        anchor_geometry["height"],
    )
    detection_diagonal = math.hypot(
        geometry["width"],
        geometry["height"],
    )

    proximity_limit = 0.70 * max(
        anchor_diagonal,
        detection_diagonal,
        1.0,
    )

    return overlap >= 0.08 or center_distance <= proximity_limit


def select_target_solder_pair(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    이미지 전체 검출 중 대상 R0805의 좌/우 납땜 Pair를 선택한다.

    먼저 좌/우 접합부를 대표할 anchor pair를 동적으로 선택한 뒤,
    각 anchor와 겹치거나 매우 가까운 추가 검출도 같은 접합부 cluster로
    포함한다. 따라서 한 접합부에 여러 클래스/mask가 겹쳐 검출되는 경우도
    결함 정보를 잃지 않는다.

    반환:
    - selected_detections: 최종 판정/Overlay에 사용할 대상 접합부 검출들
    - filtered_out_detections: 주변 부품 또는 Pair 미선택 detection
    - target_selection: Pair 선택 근거 및 디버깅 메타데이터

    Pair를 만들 수 없으면 selected_detections=[] 를 반환해
    서비스가 보수적으로 REVIEW로 이동하도록 한다.
    """

    enriched_all: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for detection in detections:
        geometry = get_bbox_geometry(detection)

        if geometry is None:
            enriched = {
                **detection,
                "target_pair_member": False,
                "filter_reason": "invalid_bbox",
            }
            enriched_all.append(enriched)
            continue

        enriched = {
            **detection,
            "center_xy": [
                round(geometry["center_x"], 2),
                round(geometry["center_y"], 2),
            ],
            "target_pair_member": False,
        }
        enriched_all.append(enriched)

        if float(detection.get("confidence", 0.0)) >= (
            PAIR_CANDIDATE_MIN_CONFIDENCE
        ):
            eligible.append(enriched)
        else:
            enriched["filter_reason"] = (
                "below_pair_candidate_confidence"
            )

    pair_candidates: list[dict[str, Any]] = []

    for first, second in combinations(eligible, 2):
        candidate = build_pair_candidate(
            left_detection=first,
            right_detection=second,
            image_width=image_width,
            image_height=image_height,
        )

        if candidate is not None:
            pair_candidates.append(candidate)

    pair_candidates.sort(key=lambda item: item["score"])

    if not pair_candidates:
        filtered = []
        for detection in enriched_all:
            if "filter_reason" not in detection:
                detection["filter_reason"] = "no_valid_target_pair"
            filtered.append(detection)

        return (
            [],
            filtered,
            {
                "mode": "dynamic_solder_pair",
                "status": "not_found",
                "candidate_detection_count": len(eligible),
                "pair_candidate_count": 0,
                "selected_anchor_ids": [],
                "selected_detection_ids": [],
                "rule": (
                    "좌/우 납땜의 수평 분리, Y 정렬, Pair 중앙성, "
                    "검출 confidence를 이용해 대상 Pair를 동적으로 선택"
                ),
            },
        )

    best = pair_candidates[0]
    left_anchor = best["left_detection"]
    right_anchor = best["right_detection"]
    left_anchor_id = int(left_anchor["detection_id"])
    right_anchor_id = int(right_anchor["detection_id"])

    selected: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    selected_ids: list[int] = []

    left_anchor_geometry = get_bbox_geometry(left_anchor)
    right_anchor_geometry = get_bbox_geometry(right_anchor)

    for detection in enriched_all:
        detection_id = int(detection["detection_id"])

        is_left_member = belongs_to_anchor_cluster(
            detection,
            left_anchor,
        )
        is_right_member = belongs_to_anchor_cluster(
            detection,
            right_anchor,
        )

        if is_left_member or is_right_member:
            # 양쪽 cluster에 동시에 가까운 드문 경우에는 더 가까운 anchor를 택한다.
            geometry = get_bbox_geometry(detection)
            pair_side = "left"

            if (
                geometry is not None
                and left_anchor_geometry is not None
                and right_anchor_geometry is not None
                and is_left_member
                and is_right_member
            ):
                left_distance = math.hypot(
                    geometry["center_x"] - left_anchor_geometry["center_x"],
                    geometry["center_y"] - left_anchor_geometry["center_y"],
                )
                right_distance = math.hypot(
                    geometry["center_x"] - right_anchor_geometry["center_x"],
                    geometry["center_y"] - right_anchor_geometry["center_y"],
                )
                pair_side = (
                    "left" if left_distance <= right_distance else "right"
                )
            elif is_right_member:
                pair_side = "right"

            selected_detection = {
                **detection,
                "target_pair_member": True,
                "pair_side": pair_side,
                "pair_anchor": detection_id in {
                    left_anchor_id,
                    right_anchor_id,
                },
                "pair_selection_score": best["score"],
            }
            selected_detection.pop("filter_reason", None)
            selected.append(selected_detection)
            selected_ids.append(detection_id)
        else:
            filtered_detection = {**detection}
            if "filter_reason" not in filtered_detection:
                filtered_detection["filter_reason"] = (
                    "not_selected_target_pair"
                )
            filtered.append(filtered_detection)

    selected.sort(
        key=lambda item: (
            0 if item.get("pair_side") == "left" else 1,
            -float(item.get("confidence", 0.0)),
        )
    )

    # 상위 후보는 디버깅에 도움이 되지만 응답이 지나치게 커지지 않도록 5개만 보존.
    top_candidates = [
        {
            "left_detection_id": int(
                item["left_detection"]["detection_id"]
            ),
            "right_detection_id": int(
                item["right_detection"]["detection_id"]
            ),
            "pair_center_xy": item["pair_center_xy"],
            "horizontal_distance_px": item["horizontal_distance_px"],
            "vertical_distance_px": item["vertical_distance_px"],
            "mean_confidence": item["mean_confidence"],
            "score": item["score"],
        }
        for item in pair_candidates[:5]
    ]

    return (
        selected,
        filtered,
        {
            "mode": "dynamic_solder_pair",
            "status": "selected",
            "candidate_detection_count": len(eligible),
            "pair_candidate_count": len(pair_candidates),
            "selected_anchor_ids": [
                left_anchor_id,
                right_anchor_id,
            ],
            "selected_detection_ids": selected_ids,
            "selected_detection_count": len(selected_ids),
            "pair_center_xy": best["pair_center_xy"],
            "pair_score": best["score"],
            "horizontal_distance_px": best["horizontal_distance_px"],
            "vertical_distance_px": best["vertical_distance_px"],
            "mean_confidence": best["mean_confidence"],
            "top_pair_candidates": top_candidates,
            "rule": (
                "좌/우 anchor Pair를 동적으로 선택한 뒤 각 anchor와 "
                "겹치거나 가까운 추가 mask를 같은 접합부 cluster로 포함"
            ),
        },
    )

def save_target_pair_overlay(
    image_rgb: np.ndarray,
    detections: list[dict[str, Any]],
    target_selection: dict[str, Any],
    output_path: Path,
) -> None:
    """
    동적으로 선택된 대상 납땜 Pair만 표시한 Overlay를 저장한다.

    주변 부품에서 모델이 검출한 객체는 최종 Overlay에 표시하지 않는다.
    """

    try:
        overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if not detections:
            cv2.putText(
                overlay,
                "Target solder pair not found - REVIEW",
                (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        centers: list[tuple[int, int]] = []

        for detection in detections:
            class_name = str(
                detection.get("class_name", "unknown")
            )
            confidence = float(
                detection.get("confidence", 0.0)
            )
            bbox = detection.get("bbox_xyxy") or []
            polygon = detection.get("polygon") or []
            center = detection.get("center_xy") or []

            if len(center) == 2:
                centers.append(
                    (
                        int(round(float(center[0]))),
                        int(round(float(center[1]))),
                    )
                )

            if polygon:
                points = np.array(
                    [
                        [int(round(x)), int(round(y))]
                        for x, y in polygon
                    ],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                mask_layer = overlay.copy()
                cv2.fillPoly(
                    mask_layer,
                    [points],
                    (0, 220, 255),
                )
                overlay = cv2.addWeighted(
                    mask_layer,
                    0.30,
                    overlay,
                    0.70,
                    0,
                )

                cv2.polylines(
                    overlay,
                    [points],
                    isClosed=True,
                    color=(0, 220, 255),
                    thickness=2,
                )

            if len(bbox) == 4:
                x1, y1, x2, y2 = [
                    int(round(float(value)))
                    for value in bbox
                ]

                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 255),
                    2,
                )

                pair_side = str(detection.get("pair_side", ""))
                side_prefix = (
                    f"{pair_side.upper()} " if pair_side else ""
                )

                label = (
                    f"{side_prefix}{class_name} "
                    f"{confidence * 100:.1f}%"
                )

                label_y = max(22, y1 - 8)

                cv2.putText(
                    overlay,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )

        if len(centers) == 2:
            cv2.line(
                overlay,
                centers[0],
                centers[1],
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            pair_center = target_selection.get("pair_center_xy") or []
            if len(pair_center) == 2:
                text_x = int(round(float(pair_center[0]))) - 85
                text_y = max(
                    28,
                    int(round(float(pair_center[1]))) - 28,
                )
                cv2.putText(
                    overlay,
                    "Target solder pair",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        saved = cv2.imwrite(
            str(output_path),
            overlay,
        )

        if not saved:
            raise RuntimeError(
                "cv2.imwrite가 False를 반환했습니다."
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Overlay 이미지 저장 실패: {exc}",
        ) from exc
