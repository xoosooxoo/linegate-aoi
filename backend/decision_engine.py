from __future__ import annotations

from typing import Any


DEFECT_CLASSES = {
    "exc_solder",
    "poor_solder",
    "spike",
}


def determine_solder_decision(
    detections: list[dict[str, Any]],
    confidence_threshold: float = 0.70,
) -> dict[str, str]:
    """
    V2/V2.1 납땜 검사 결과를
    RELEASE / HOLD / REVIEW로 변환한다.

    주의:
    - AI 모델은 납땜 상태를 예측한다.
    - RELEASE/HOLD/REVIEW는 서비스 운영 규칙이다.
    """

    if not detections:
        return {
            "status": "REVIEW",
            "reason": "검출 결과 없음",
            "route": "이미지 품질, 촬영 조건 및 모델 입력을 확인하세요.",
        }

    confident_detections = [
        detection
        for detection in detections
        if detection["confidence"] >= confidence_threshold
    ]

    if not confident_detections:
        return {
            "status": "REVIEW",
            "reason": "모든 검출 결과가 신뢰도 기준 미달",
            "route": "숙련 검사자의 추가 검토가 필요합니다.",
        }

    confident_classes = {
        detection["class_name"]
        for detection in confident_detections
    }

    detected_defects = confident_classes & DEFECT_CLASSES

    if detected_defects:
        defect_text = ", ".join(sorted(detected_defects))

        return {
            "status": "HOLD",
            "reason": f"납땜 결함 검출: {defect_text}",
            "route": get_reinspection_route(detected_defects),
        }

    if confident_classes == {"good"}:
        return {
            "status": "RELEASE",
            "reason": "신뢰도 기준을 충족한 정상 납땜부 검출",
            "route": "가용 검사 결과상 다음 생산 진행을 권고합니다.",
        }

    return {
        "status": "REVIEW",
        "reason": "판정할 수 없는 클래스 조합",
        "route": "숙련 검사자가 결과를 확인해야 합니다.",
    }


def get_reinspection_route(defect_classes: set[str]) -> str:
    """검출 클래스에 따라 후속 확인 경로를 제안한다."""

    routes: list[str] = []

    if "poor_solder" in defect_classes:
        routes.append("납땜 부족 여부 재검 및 보강 가능성 검토")

    if "exc_solder" in defect_classes:
        routes.append("과납과 인접부 접촉 가능성 점검")

    if "spike" in defect_classes:
        routes.append("돌출 형상과 주변 간섭 가능성 점검")

    if not routes:
        return "납땜 상태를 재검하세요."

    return " / ".join(routes)