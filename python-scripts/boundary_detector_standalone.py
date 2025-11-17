#!/usr/bin/env python3
"""
Boundary Detection Standalone Script
DocAligner를 사용한 노트/문서 경계 검출 (Java 연동용)
"""

import sys
import json
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# DocAligner import
from docaligner import DocAligner


class BoundaryDetector:
    """문서 경계 검출 클래스"""

    def __init__(self):
        """DocAligner 모델 초기화"""
        self.model = DocAligner(
            model_type='heatmap',
            model_cfg='fastvit_sa24',
            backend='cpu'
        )

    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        DocAligner를 사용한 코너 검출

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            4개의 코너 포인트 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or None
        """
        # RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 모델 실행
        corners = self.model(image_rgb)

        if corners is None or len(corners) != 4:
            return None

        return corners

    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        코너 포인트를 표준 순서로 정렬
        순서: [top-left, top-right, bottom-right, bottom-left]
        """
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]

        # top-left부터 시작
        sums = sorted_corners[:, 0] + sorted_corners[:, 1]
        min_idx = np.argmin(sums)
        ordered = np.roll(sorted_corners, -min_idx, axis=0)

        return ordered

    def draw_boundary(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """검출된 경계를 이미지에 그리기"""
        result = image.copy()
        corners_int = np.int32(corners)

        # 사각형 그리기
        cv2.drawContours(result, [corners_int], -1, (0, 255, 0), 3)

        # 코너에 원 그리기
        for i, corner in enumerate(corners_int):
            cv2.circle(result, tuple(corner), 8, (0, 0, 255), -1)
            cv2.putText(result, str(i + 1), tuple(corner + [10, -10]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 정보 텍스트
        cv2.putText(result, "Boundary detected - DocAligner", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return result

    def process_image(self, image_path: str, output_dir: str = "output") -> dict:
        """
        이미지 처리 및 결과 반환

        Args:
            image_path: 입력 이미지 경로
            output_dir: 출력 디렉토리

        Returns:
            결과 딕셔너리 (JSON 변환용)
        """
        # 이미지 읽기
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }

        image = cv2.imread(image_path)
        if image is None:
            return {
                "success": False,
                "error": f"Cannot read image: {image_path}"
            }

        # 코너 검출
        corners = self.detect_corners(image)

        if corners is None:
            return {
                "success": False,
                "error": "Failed to detect document corners"
            }

        # 코너 정렬
        ordered_corners = self.order_corners(corners)

        # 시간 기반 디렉토리 생성
        now = datetime.now()
        time_str = now.strftime("%m%d_%H%M")  # 월일_시분
        base_name = Path(image_path).stem

        # 실행별 디렉토리 생성: imageName_1025_1504
        result_dir = os.path.join(output_dir, f"{base_name}_{time_str}")
        os.makedirs(result_dir, exist_ok=True)

        # 경계 그리기
        boundary_image = self.draw_boundary(image, ordered_corners)
        boundary_path = os.path.join(result_dir, "boundary.jpg")
        cv2.imwrite(boundary_path, boundary_image)

        # 좌표 JSON 저장
        coords_json_path = os.path.join(result_dir, "result.json")
        coords_data = {
            "imagePath": image_path,
            "timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
            "method": "DocAligner",
            "corners": [
                {"index": i + 1, "x": float(corner[0]), "y": float(corner[1])}
                for i, corner in enumerate(ordered_corners)
            ]
        }
        with open(coords_json_path, 'w', encoding='utf-8') as f:
            json.dump(coords_data, f, ensure_ascii=False, indent=2)

        # 성공 결과 반환
        return {
            "success": True,
            "corners": ordered_corners.tolist(),
            "boundaryImage": boundary_path,
            "resultJson": coords_json_path,
            "resultDirectory": result_dir,
            "timestamp": now.isoformat(),
            "imageShape": list(image.shape)
        }


def main():
    """메인 실행 함수 (Java에서 호출용)"""
    # 커맨드라인 인자 확인
    if len(sys.argv) < 2:
        result = {
            "success": False,
            "error": "Usage: python3 boundary_detector_standalone.py <image_path> [output_dir]"
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    try:
        # 검출기 생성
        detector = BoundaryDetector()

        # 이미지 처리
        result = detector.process_image(image_path, output_dir)

        # JSON 출력 (Java가 읽을 것)
        print(json.dumps(result, ensure_ascii=False))

        # 성공 시 exit code 0, 실패 시 1
        sys.exit(0 if result["success"] else 1)

    except Exception as e:
        # 예외 발생 시
        error_result = {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
