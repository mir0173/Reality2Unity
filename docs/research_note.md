## 1. 과거 연구 단일 사진 복원 재현
MiDaS DPT_Large로 단일 사진의 상대 깊이를 추정하고, 2024년 보고서의 높이값 정규화와 상단 시점 변환을 재현했다. 이후 상대 깊이와 가정한 초점거리를 이용해 이미지 픽셀을 3D 좌표로 역투영하고, 원본 RGB를 붙여 `pointcloud_baseline.ply`로 저장했다. 

## 2. ETH3D Terrace 다중 시점 복원
Drive의 `Reality2Unity/multiview/images`에 있는 Terrace 사진 23장으로 COLMAP SfM을 실행했다. 사진 **23장 전부가 등록**됐고, 희소점 **9,068개**, 평균 재투영 오차 **1.24654px**를 기록했다. 


| 결과 파일 | PatchMatch 최대 해상도 | Fusion 최대 해상도 |
|---|---:|---:|
| `dense_multiview.ply` | 1200px | 1200px |
| `dense_multiview_1600.ply` | 1600px | 1200px |
PatchMatch 해상도를 높인 뒤에도 일부 벽면과 주변 표면이 복원되지 않았다.

## 3. 메시 생성 실험과 실행 환경 문제
- **Poisson**에서 trim 10, 5, 2를 비교했다. 
- **Delaunay**는 COLMAP CLI 3.9.1에서 `Integrating image` 단계의 첫 이미지 처리 중 진행이 멈춰 중단했다. 
- **Colab 환경**에서 `pycolmap-cuda12`와 PyTorch CUDA 13 패키지 사이의 의존성 충돌을 겪었다. 

## 4. Python 중심 구조로 전환
기존 실험은 `baseline_colab.ipynb`에 남겨두고, 새 `notebooks/multiview_runner.ipynb`를 추가했다. 
