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
python3 make_inat_image_stratified_family.py \
  --inat_root "data/inat2017_raw" \
  --train_json "data/inat2017_raw/train2017.json" \
  --val_json   "data/inat2017_raw/val2017.json" \
  --cat_to_family_json "data/inat2017_meta_family/category_to_family.json" \
  --out_npz "data/npz/inat2017_images_family_strat_t50k_c30k_te10k_seed1.npz" \
  --seed 1 \
  --image_size 224 \
  --n_train 50000 \
  --n_calib 30000 \
  --n_test 10000 \
  --min_family_count 250 \
  --min_test_per_family 1 \
  --min_calib_per_family 1 \
  --save_indices
```

### 생성 파일
iNaturalist 2017 데이터로부터 family-level label 기준의 stratified split을 생성한다. 생성 파일은 다음을 포함한다.
- `X_train`, `X_cal`, `X_test` : 이미지 데이터
- `y_train`, `y_cal`, `y_test` : 정답 레이블
- `idx_train`, `idx_calib`, `idx_test` (Optional) : 전체 pool 기준 인덱스
- `meta` : family 매핑 정보 및 데이터 생성 설덩

### 참고
- `--min_family_count` 보다 수가 적은 family는 사전에 제거됨.
- 모든 split은 family 분포를 유지하도록 stratified sampling됨.


---

## 3. 모델 학습 및 확률 NPZ 생성
앞서 변환한 이미지 NPZ 파일을 input으로 하고, 확률 NPZ 를 output으로 하는 모델을 학습한다. 이 단계의 목적은 각 이미지에 대한 class probability를 얻는 것이며, 모델의 분류 정확도 자체는 Conformal prediction 방법 특성상 크게 중요하지 않다. 그러나 정확도가 너무 낮은 모델의 경우에는 class probability가 거의 동일하여 자칫 prediction set size를 너무 크게 만들 수 있으므로, 정확도가 너무 낮은 모델을 사용하는 것에는 주의가 필요하다. 

### 실행 예시 (ResNet50, head-only)
```bash
python3 scripts/train_and_export_probs_inat.py \
  --img_npz data/npz/inat2017_images_strat_t50k_v30k_te10k_seed1.npz \
  --out_prob_npz data/npz/inat2017_probs_strat_selA_calB_test_rn50_head_t50k_ep10_seed1.npz \
  --model resnet50 \
  --finetune head \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --calib_split_seed 1 \
  --seed 1
```


### 출력 파일
```bash
data/npz/
└── inat2017_probs_selA_calB_test_rn50_head_t50k_ep10_seed1.npz
```
이 확률 NPZ 파일은 이후의 Conformal Prediction 방법들의 input으로 사용된다. 

---

## 4. Conformal Prediction

### 비교 방법
- **GCP (Global Conformal Prediction)**  
  A baseline conformal prediction method using a global threshold  
  [Vovk et al., 2005; Angelopoulos et al., 2022].

- **CC-CP (Class-Conditional Conformal Prediction)**  
  Class-wise conformal thresholds  
  [Ding et al., 2023].

- **SCCP (Shrinkage-Clustered Conformal Prediction)**  
  Clustered and shrinkage-based conformal thresholds (ours).

---

## 5. SCCP 
### 5.1 Class Embedding
- calibration data에서 각 클래스별 score 수집
- score quantile embedding (Simulation) or mean embedding (iNat2017)
### 5.2 Clustering
- embedding vector에 대해 kmeans clustering 적용
- `Kc` : Cluster number (default = 10)
### 5.3 Shrinkage
- cluster 단위로 shrinkage parameter `lambda_hat` 학습
- tail class에 대한 prediction set size를 줄이는 것이 목적

### 실행 예시
```bash
# (1) Softmax score로 CP 수행 + SCCP 클러스터링은 logit 기반 임베딩 사용
python scripts/run_cp_from_npz.py \
  --npz data/npz/inat2017_probs_strat_selA_calB_test_rn50_head_t50k_ep10_seed1.npz \
  --K 5089 --alpha 0.1 --seed 1 \
  --score softmax \
  --emb_source logit \
  --weighted_kmeans \
  --clusters 10 --tau 50 --beta 0.5 \
  --tail_mode npz --tail_frac 0.2
```
```bash
# (2) 위와 동일 + CCCP(Ding) 결과도 같이 출력
python scripts/run_cp_from_npz.py \
  --npz data/npz/inat2017_probs_strat_selA_calB_test_rn50_head_t50k_ep10_seed1.npz \
  --K 5089 --alpha 0.1 --seed 1 \
  --score softmax \
  --emb_source logit \
  --weighted_kmeans \
  --clusters 10 --tau 50 --beta 0.5 \
  --tail_mode npz --tail_frac 0.2 \
  --run_cccp \
  --cccp_gamma 0.5 --cccp_M 10
```
```bash
# (3) APS / RAPS도 같은 설정으로 비교 (score만 바꿔서 반복)
python scripts/run_cp_from_npz.py \
  --npz data/npz/inat2017_probs_strat_selA_calB_test_rn50_head_t50k_ep10_seed1.npz \
  --K 5089 --alpha 0.1 --seed 1 \
  --score aps \
  --emb_source logit \
  --weighted_kmeans \
  --clusters 10 --tau 50 --beta 0.5 \
  --tail_mode npz --tail_frac 0.2

python scripts/run_cp_from_npz.py \
  --npz data/npz/inat2017_probs_strat_selA_calB_test_rn50_head_t50k_ep10_seed1.npz \
  --K 5089 --alpha 0.1 --seed 1 \
  --score raps --raps_lambda 0.05 --raps_kreg 5 \
  --emb_source logit \
  --weighted_kmeans \
  --clusters 10 --tau 50 --beta 0.5 \
  --tail_mode npz --tail_frac 0.2
```
#### Option
- `--score softmax|aps|raps` : 예측집합을 만드는 CP score
- `--emb_source logit` : SCCP에서 클래스 클러스터링 임베딩만 logit 기반으로 생성
- `--weighted_kmeans` : 클래스별 표본수에 비례한 가중치로 K-means
- `--clusters Kc, --tau, --beta` : SCCP 하이퍼파라미터
---

## 6. 출력 및 평가 지표
기본 출력 지표는 다음과 같다. 
- Marginal coverage
- Average set size
- Coverage gap
- Tail/Head coverage
- Tail/Head set size
---



# References
- Vovk, V., Gammerman, A., and Shafer, G. (2005).
*Algorithmic Learning in a Random World*.
Springer.

- Angelopoulos, A. N., Bates, S., Malik, J., and Jordan, M. I. (2022).
*Uncertainty Sets for Image Classifiers using Conformal Prediction*.
Advances in Neural Information Processing Systems (NeurIPS 2022).

- Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C.,
  Shepard, A., Adam, H., Perona, P., and Belongie, S. (2018).
*The iNaturalist Species Classification and Detection Dataset*.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018).
