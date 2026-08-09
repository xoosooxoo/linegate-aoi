from __future__ import annotations

import io
import time
import uuid
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
# ROI 설정
# ---------------------------------------------------------
# PoC 기본값:
# - 이미지 전체를 추론하되,
# - 아래 ROI 안에 중심점이 들어오는 검출만 최종 판정에 사용한다.
#
# 좌표는 픽셀 고정값이 아니라 이미지 크기에 대한 비율이므로
# 해상도가 달라도 동일한 비율로 적용된다.
#
# 형식: (x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio)
#
# 실제 V2 / V2.1 샘플에서 대상 R0805가 더 좁은 영역에 항상 위치한다면
# 아래 값을 더 타이트하게 조정하면 주변 False Positive를 더 줄일 수 있다.
ROI_RATIOS = {
    "V2": (0.10, 0.15, 0.90, 0.85),
    "V2.1": (0.10, 0.15, 0.90, 0.85),
}


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
    version="0.1.0",
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
    - 검출 클래스
    - confidence
    - bounding box
    - polygon
    - mask 면적
    - RELEASE/HOLD/REVIEW
    - Overlay 이미지 URL
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
            # 매우 낮은 confidence 결과도 먼저 받아야
            # 이후 REVIEW 규칙을 적용할 수 있다.
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

    # 2) 검사 대상 ROI 내부 검출만 최종 판정에 사용
    roi = build_roi(
        view=normalized_view,
        image_width=pil_image.width,
        image_height=pil_image.height,
    )

    detections, filtered_out_detections = filter_detections_by_roi(
        detections=raw_detections,
        roi=roi,
    )

    # 3) ROI 안의 검출만 RELEASE / HOLD / REVIEW 판정에 반영
    decision = determine_solder_decision(
        detections=detections,
        confidence_threshold=confidence_threshold,
    )

    overlay_filename = f"{request_id}.jpg"
    overlay_path = OVERLAY_DIR / overlay_filename

    # YOLO result.plot()을 그대로 쓰면 ROI 밖 검출도 보이므로,
    # 필터링된 detection만 표시하는 Overlay를 별도로 생성한다.
    save_filtered_overlay(
        image_rgb=image_rgb,
        detections=detections,
        roi=roi,
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
        "roi": {
            "x_min": roi["x_min"],
            "y_min": roi["y_min"],
            "x_max": roi["x_max"],
            "y_max": roi["y_max"],
            "ratio": ROI_RATIOS[normalized_view],
            "rule": "검출 중심점이 ROI 안에 있을 때만 최종 판정에 사용",
        },
        "inference": {
            "time_ms": round(inference_time_ms, 2),
            "confidence_threshold": confidence_threshold,
        },
        "detections": detections,
        "filtered_out_detections": filtered_out_detections,
        "summary": {
            "raw_detection_count": len(raw_detections),
            "total_detection_count": len(detections),
            "filtered_out_detection_count": len(filtered_out_detections),
            "confident_detection_count": len(confident_detections),
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
            "ROI 밖 검출은 최종 판정 및 Overlay에서 제외됩니다."
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



def build_roi(
    view: str,
    image_width: int,
    image_height: int,
) -> dict[str, int]:
    """View별 비율 설정을 실제 픽셀 ROI로 변환한다."""

    if view not in ROI_RATIOS:
        raise HTTPException(
            status_code=400,
            detail=f"ROI 설정이 없는 View입니다: {view}",
        )

    x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio = ROI_RATIOS[view]

    return {
        "x_min": int(round(image_width * x_min_ratio)),
        "y_min": int(round(image_height * y_min_ratio)),
        "x_max": int(round(image_width * x_max_ratio)),
        "y_max": int(round(image_height * y_max_ratio)),
    }


def get_detection_center(
    detection: dict[str, Any],
) -> tuple[float, float]:
    """
    Detection의 대표 중심점을 구한다.
    Polygon이 있으면 polygon 평균좌표를 사용하고,
    없으면 bbox 중심점을 사용한다.
    """

    polygon = detection.get("polygon") or []

    if polygon:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]

        return (
            sum(xs) / len(xs),
            sum(ys) / len(ys),
        )

    bbox = detection.get("bbox_xyxy") or []

    if len(bbox) == 4:
        x1, y1, x2, y2 = map(float, bbox)

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

    return (-1.0, -1.0)


def filter_detections_by_roi(
    detections: list[dict[str, Any]],
    roi: dict[str, int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    검출 중심점이 ROI 안에 있는 detection만 유효 결과로 남긴다.

    반환:
    - valid_detections: 최종 판정에 사용
    - filtered_out_detections: ROI 밖이라 제외된 검출
    """

    valid_detections: list[dict[str, Any]] = []
    filtered_out_detections: list[dict[str, Any]] = []

    for detection in detections:
        center_x, center_y = get_detection_center(detection)

        is_inside = (
            roi["x_min"] <= center_x <= roi["x_max"]
            and roi["y_min"] <= center_y <= roi["y_max"]
        )

        enriched_detection = {
            **detection,
            "center_xy": [
                round(center_x, 2),
                round(center_y, 2),
            ],
            "inside_roi": is_inside,
        }

        if is_inside:
            valid_detections.append(enriched_detection)
        else:
            filtered_out_detections.append(enriched_detection)

    return valid_detections, filtered_out_detections


def save_filtered_overlay(
    image_rgb: np.ndarray,
    detections: list[dict[str, Any]],
    roi: dict[str, int],
    output_path: Path,
) -> None:
    """
    ROI와 ROI 내부 detection만 표시한 Overlay를 저장한다.

    ROI 밖에서 모델이 검출한 객체는 화면에도 표시하지 않는다.
    """

    try:
        # OpenCV 저장/그리기용 BGR 변환
        overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 검사 ROI 표시
        cv2.rectangle(
            overlay,
            (roi["x_min"], roi["y_min"]),
            (roi["x_max"], roi["y_max"]),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            overlay,
            "Inspection ROI",
            (roi["x_min"], max(25, roi["y_min"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for detection in detections:
            class_name = str(
                detection.get("class_name", "unknown")
            )
            confidence = float(
                detection.get("confidence", 0.0)
            )
            bbox = detection.get("bbox_xyxy") or []
            polygon = detection.get("polygon") or []

            # Polygon mask/contour 표시
            if polygon:
                points = np.array(
                    [
                        [int(round(x)), int(round(y))]
                        for x, y in polygon
                    ],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                # 반투명 mask
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

            # Bounding box 표시
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

                label = (
                    f"{class_name} "
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
