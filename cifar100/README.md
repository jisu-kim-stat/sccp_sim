- train_cifar100.py : balanced dataset 

# LT Dataset
- train_cifar100_lt.py : Long tail dataset (train/cal is LT, test is raw)
## 실행 코드 ##
```bash
python train_cifar100_lt.py \
  --data_root ./data \
  --out_dir ./out/cifar100LT_probs \
  --use_gpu \
  --lt --imb_type exp --imb_factor 100 \
  --split_mode fracs \
  --train_frac 0.6 --select_frac 0.2 --calib_frac 0.2 \
  --epochs 100 --batch_size 128 --lr 0.01 \
  --seed_data 1 --seed_train 1
  --arch resnet152 #["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]
```

## tau sweep ##
```bash
python run_cp_from_npz.py \
  --npz "/home/jisukim/sccp_sim/cifar100/out/cifar100LT_probs/cifar100_resnet50_e100_bs128_LT_exp_IF100_tailfrac0.2.npz" \
  --alpha 0.1 --K 100 --seed 1 \
  --sweep \
  --tau_list 0,1,5,10,20,50,100,200 \
  --clusters_list 5,10,20 \
  --embed score_quantile
```
- 위 sweep을 통해 가장 잘 나온 tau 확인 후 아래 과정 진행

## Perform CP using tau ##
```bash
python run_cp_from_npz.py \
  --npz "/home/jisukim/sccp_sim/cifar100/out/cifar100LT_probs/cifar100_resnet101_e100_bs128_LT_exp_IF100_tailfrac0.2.npz"
  --alpha 0.1 \
  --K 100 \
  --clusters 10 \
  --tau 10 \
  --seed 1 \
  --out results_cifar100LT_resnet152_tau10.json
```


## To latex table ##
```bash
python scripts/json_to_latex_table.py \
  --json %앞에서 설정한 json 파일명 \
  --single_block \
  --caption "CIFAR-100-LT (IF=100), ResNet-18 (100 epochs)." \
  --label "tab:cifar100lt_resnet18_e100_tau5" \ #모델명, tau 수정 
  --out_tex tables/cifar100lt_resnet18_e100_tau5.tex #모델명, tau 수정 


- train_cifar100_imbworld.py : Long tail dataset with train/cal/test is all LT
- tarin_cifar100_scc_raps.py : balanced dataset with SCCP_RAPS method (not finished yet. need to fix)
