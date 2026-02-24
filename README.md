# Shrinkage-Class-Clustered Conformal Prediction (SCC-CP)
본 저장소는 대규모 다중 클래스 이미지 분류 문제에서 **Global CP / Class-Conditional CP / Shrinkage Class-Clustered CP (SCCP)** 방법을 비교하기 위한 실험 코드 모음이다.

주요 실험 데이터셋은 다음과 같다.
- iNaturalist (inat2017) 
- CIFAR100 (cifar100)
- SDSS (Photometric Redshift) 

---

## 전체 파이프라인 요약

1. 이미지 데이터 → NPZ 변환  
2. 이미지 분류 모델 학습 및 **확률 출력 NPZ 생성**  
3. 확률 NPZ를 입력으로 conformal prediction 수행  
4. GCP / CCCP / SCCP 성능 비교 (coverage, set size 등)

---

## 0. Dependencies

The following Python packages are required to run the experiments:

- Python ≥ 3.9
- numpy
- scipy
- scikit-learn
- pandas
- matplotlib
- torch
- torchvision
- tqdm
- pillow
- tensorflow
- tensorflow-datasets

## 1. Dataset
Experiments are conducted on the iNaturalist 2017 (iNat2017) dataset,
a large-scale and highly imbalanced real-world image classification benchmark
[Van Horn et al., 2018].

iNat2017은 실제 자연 환경에서 수집된 대규모 이미지 분류 데이터셋으로, 클래스 간 샘플 수의 불균형이 매우 심한것이 특징이다. 일부 소수 클래스는 극히 적은 관측치를 가지지만, 다수의 클래스는 수천장 이상의 이미지를 포함하고 있어 long-tailed 분포를 잘 반영한다. 

이러한 특성으로 인해 GCP가 소수 클래스에 대해 과도하게 보수적이거나, 반대로 신뢰성을 보장하지 못하는 경우를 드러내기에 적합한 데이터라 생각된다. 본 실험에서는 iNat2017을 활용하여 클래스 불균형이 심한 상황에서도 우리가 제안하는 SCCP 방법이 기존 방법들에 비해 안정적이고 균형잡힌 prediction set을 제공할 수 있음을 검증하고자 한다. 


## 2. iNat2017 데이터를 NPZ로 변환
원본 파일은 용량 문제로 직접 사용하지 않으며, 실험에 필요한 train/val/test split을 포함한 NPZ 파일로 변환하여 사용한다. 

### 실행 예시
```bash
python3 scripts/make_inat_image_stratified_family.py \
  --data_dir "data/tfds" \
  --pool_split "train+validation" \
  --cat_to_family_json "data/inat2017_meta_family/category_to_family.json" \
  --out_npz "data/npz/inat2017_images_family_dingstyle_seed1.npz" \
  --seed 1 \
  --image_size 224 \
  --min_family_count 250 \
  --train_frac 0.7 \
  --sel_frac 0.1 \
  --cal_frac 0.1 \
  --test_frac 0.1 \
  --save_indices
```

### 생성 파일
iNaturalist 2017 데이터로부터 family-level label 기준의 stratified split을 생성한다.
Species-level 라벨을 family-level로 변환한 뒤, `train+validation` 풀을 집계해서 각 family별로 동일한 비율을 갖도록 분할한다.
생성된 NPZ 파일에는 다음이 포함된다.
- 이미지 데이터
  -  `X_train` : 모델 학습용 이미지
  -  `X_sel` : selection (cluster / hyperparameter 선택)용 이미지
  -  `X_cal` : conformal calibration용 이미지
  -  `X_test` : 평가용 이미지
- 정답 label (family-level) : `y_train, y_sel, y_cal, y_test`
- 인덱스 정보 (Optional) : `idx_train, idx_sel, idx_cal, idx_test `
  (train+validation pool 기준의 인덱스)
- 메타데이터
  - `meta` : 
    - species -> family 매핑 정보
    - 사용된 split 비율
    - 필터링된 family 조건
    - 데이터 생성 설정 (seed, image size 등)

### 참고
- Ding et al. (2025)와 비슷하게 split한다 : calibration과 test가 동일한 분포를 갖도록 설계되어, conformal prediction의 exchangeability 가정을 만족한다.
- Family-level split을 사용함으로써, species-level의 극단적인 long-tail로 인한 불안정성을 완화한다.
- `--save_indices` 옵션을 사용하면 동일한 pool 기준에서 실험 재현이 가능하다.

---

## 3. 모델 학습 및 확률 NPZ 생성
앞서 변환한 이미지 NPZ 파일을 입력으로 사용하여, 각 이미지에 대한 class probability를 산출하는 분류 모델을 학습한다.
이 단계의 핵심 목적은 Conformal Prediction에 사용될 확률 벡터(probability vector)를 생성하는 것이며,
모델의 분류 정확도 자체는 Conformal Prediction 방법의 특성상 최우선 목표는 아니다.

다만, 모델의 성능이 지나치게 낮을 경우 모든 클래스에 대해 확률이 거의 균등하게 분포하게 되어
prediction set의 크기가 불필요하게 커질 수 있다.
따라서 확률 분포가 의미 있게 분리될 수 있을 정도의 최소한의 분류 성능은 확보되어야 하며,
정확도가 극단적으로 낮은 모델을 사용하는 것은 지양한다.

본 실험에서는 사전학습된 ImageNet 가중치를 초기값으로 사용하는 표준 CNN 모델을 사용하여
위 목적에 부합하는 class probability를 생성한다.

### 실행 예시 (ResNet50)
```bash
python3 scripts/train_and_export_probs_inat_family.py \
  --img_npz data/npz/inat2017_images_family_dingstyle_seed1.npz \
  --out_npz data/npz/inat2017_probs_family_resnet50_seed1.npz \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-4 \
  --finetune full \
  --num_workers 8 \
  --amp \
  --seed 1

```


### 출력 파일
```bash
data/npz/
└── inat2017_probs_family_resnet50_seed1.npz
```
해당 확률 NPZ 파일은 이후 단계에서 수행되는
Conformal Prediction 및 변형된 CP 방법들의 입력 데이터로 사용된다.
파일에는 selection, calibration, test split에 대한 class probability, label,
그리고 필요 시 score 계산을 위한 logits 정보가 함께 저장된다.

---

## 4. Conformal Prediction
본 단계에서는 앞서 학습된 분류 모델로부터 얻은 class probability를 입력으로 하여
Conformal Prediction 방법을 적용하고, 예측 집합(prediction set)의 특성을 평가한다.
단순히 marginal coverage를 맞추는 것뿐 아니라,
class imbalance 환경에서 tail class에서의 과도한 prediction set 확장(set size inflation)을 완화하는 방법을 비교·분석하는 것을 목적으로 한다.

이를 위해 다음 세 가지 방법을 비교한다.

- LCCP (classwise): 클래스별 calibration만 사용하는 class-conditional conformal prediction

- CCCP (clusterwise): 클래스들을 score 분포 유사도 기반으로 클러스터링하여, cluster 단위로 calibration을 공유하는 방법

- SCCP (class–cluster shrinkage): classwise 정보와 clusterwise 정보를 shrinkage로 결합하는 방법. 

클러스터링은 calibration 데이터에서 얻은 true-label score의 empirical quantile 벡터(quantile embedding)를 사용하고, k-means로 클래스 클러스터를 구성한다.


### 실행 예시
```bash
python3 scripts/run_sccp_class_cluster.py \
  --npz data/npz/inat2017_probs_family_resnet50_seed1.npz \
  --K 173 \
  --alpha 0.1 \
  --M 25 \
  --n_clusters 20 \
  --cluster_seed 1 \
  --tau_grid "0,0.5,1,2,5,10,20,50,100,200" \
  --tail_frac 0.2 \
  --tune_eps 0.01 \
  --print_tau_table \
  --out_json out/results/sccp_class_cluster_seed1.json
```

### 출력 및 평가 지표
스크립트는 TEST set에서 LCCP / CCCP / SCCP 각각에 대해 아래 지표를 출력한다.

- Marginal coverage

- Average prediction set size

- Average class-wise coverage (avg_class_cov)

- Worst-case class coverage (worst_class_cov)

- Tail / Head coverage (cov_tail, cov_head)

- Tail / Head prediction set size (size_tail, size_head)

- (참고) class-wise coverage 분산: std_class_cov, 그리고 deviation 지표: covgap, maxgap

또한 selection set에서 τ 후보별 성능 요약 테이블을 출력하며(--print_tau_table),
제약을 만족하는 후보 중 목적함수에 따라 best_tau를 선택한다.

# References
- Vovk, V., Gammerman, A., and Shafer, G. (2005).
*Algorithmic Learning in a Random World*.
Springer.

- Ding, T., Fermanian, J.-B., and Salmon, J. (2025).
*Conformal Predcition for Long-Tailed Classification*.
arXiv preprint.

- Angelopoulos, A. N., Bates, S., Malik, J., and Jordan, M. I. (2022).
*Uncertainty Sets for Image Classifiers using Conformal Prediction*.
Advances in Neural Information Processing Systems (NeurIPS 2022).

- Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C.,
  Shepard, A., Adam, H., Perona, P., and Belongie, S. (2018).
*The iNaturalist Species Classification and Detection Dataset*.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018).
