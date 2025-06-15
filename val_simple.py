#!/usr/bin/env python
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

"""
KAIST Multispectral Pedestrian Detection을 위한 간소화된 검증 스크립트
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_json(predn, jdict, path, index, class_map):
    """
    JSON 형식으로 예측 결과 저장
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        if p[4] < 0.001:  # 낮은 confidence 필터링
            continue
        jdict.append({
            "image_name": image_id,
            "image_id": int(index),
            "category_id": class_map[int(p[5])],
            "bbox": [round(x, 3) for x in b],
            "score": round(p[4], 5),
        })


def process_batch(detections, labels, iouv):
    """
    예측과 실제 라벨 비교하여 정확도 계산
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    rgbt=False,  # RGBT 입력 사용 여부
    epoch=None,
):
    # 모델 초기화/로드 및 장치 설정
    training = model is not None
    if training:  # train.py에서 호출된 경우
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # 모델 장치 가져오기
        half &= device.type != "cpu"  # CPU에서는 half precision 지원 안함
        model.half() if half else model.float()
    else:  # 직접 호출된 경우
        device = select_device(device, batch_size=batch_size)

    # 디렉토리
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 실행 증가
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 디렉토리 생성

    # 모델 로드
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # 이미지 크기 확인
    half = model.fp16  # FP16 지원 여부
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py 모델은 기본적으로 batch-size 1
            LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

    # 데이터
    data = check_dataset(data)  # 데이터셋 확인

    # 설정
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO 데이터셋
    nc = 1 if single_cls else int(data["nc"])  # 클래스 수
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # IoU 벡터 (mAP@0.5:0.95)
    niou = iouv.numel()

    # 데이터로더
    if not training:
        if pt and not single_cls:  # 모델이 데이터셋에 맞게 학습되었는지 확인
            ncm = model.model.nc
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        
        # 모델 웜업
        if isinstance(imgsz, int):
            imgsz = [imgsz, imgsz]  # 정사각형 추론
        
        # RGBT 입력 처리를 위한 웜업 수정
        if rgbt:
            # RGB + Thermal 입력을 위한 웜업
            im = torch.zeros(1, 3, imgsz[0], imgsz[1], device=device)  # RGB 이미지용
            thermal = torch.zeros(1, 3, imgsz[0], imgsz[1], device=device)  # Thermal 이미지를 3채널로 확장
            model([im, thermal])  # 리스트로 전달하여 웜업
        else:
            # 일반 RGB 입력을 위한 웜업
            model.warmup(imgsz=(1, 3, imgsz[0], imgsz[1]))
        
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # 벤치마크를 위한 정사각형 추론
        task = task if task in ("train", "val", "test") else "val"  # train/val/test 이미지 경로
        
        # 데이터로더 생성
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
            rgbt_input=rgbt,  # RGBT 입력 활성화
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # 클래스 이름 가져오기
    if isinstance(names, (list, tuple)):  # 이전 형식
        names = dict(enumerate(names))
    class_map = list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # 프로파일링 시간
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # 진행 표시줄
    
    for batch_i, (ims, targets, paths, shapes, indices) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if rgbt:  # RGBT 입력 처리
                ims = [im.to(device, non_blocking=True).float() / 255 for im in ims]  # RGB-T 입력
                nb, _, height, width = ims[0].shape  # 배치 크기, 채널, 높이, 너비
                if half:
                    ims = [im.half() for im in ims]
            else:  # 일반 RGB 입력 처리
                ims = ims.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
                nb, _, height, width = ims.shape  # 배치 크기, 채널, 높이, 너비
                if half:
                    ims = ims.half()
            
            targets = targets.to(device)

        # 추론
        with dt[1]:
            preds, train_out = model(ims) if compute_loss else (model(ims, augment=augment), None)

        # 손실
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # 픽셀로 변환
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # 자동 라벨링용
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # RGBT 입력 처리
        if rgbt:
            ims = ims[0]  # thermal 이미지

        # 메트릭
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # 라벨 수, 예측 수
            path, shape = Path(paths[si]), shapes[si][0]
            index = indices[si]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # 초기화
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                if plots:
                    confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # 예측
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(ims[si].shape[1:] if rgbt else ims.shape[2:], predn[:, :4], shape, shapes[si][1])  # 원본 공간 예측

            # 평가
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # 타겟 박스
                scale_boxes(ims[si].shape[1:] if rgbt else ims.shape[2:], tbox, shape, shapes[si][1])  # 원본 공간 라벨
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # 원본 공간 라벨
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # 저장/로그
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, index, class_map)  # COCO-JSON 사전에 추가
            callbacks.run("on_val_image_end", pred, predn, path, names, ims[si] if rgbt else ims[si])

        # 이미지 플롯
        if plots and batch_i < 3:
            desc = f"val_batch{batch_i}" if epoch is None else f"val_epoch{epoch}_batch{batch_i}"
            plot_images(ims, targets, paths, save_dir / f"{desc}_labels.jpg", names)  # 라벨
            plot_images(ims, output_to_target(preds), paths, save_dir / f"{desc}_pred.jpg", names)  # 예측

        callbacks.run("on_val_batch_end", batch_i, ims, targets, paths, shapes, preds)

    # 메트릭 계산
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # numpy로 변환
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        
        # ignore 클래스 필터링
        lbls = stats[3].astype(int)
        lbls = lbls[lbls >= 0]  # ignore 클래스(-1) 제외
        nt = np.bincount(lbls, minlength=nc)  # 클래스별 타겟 수
    else:
        nt = torch.zeros(1)

    # 결과 출력
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # 출력 형식
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # 클래스별 결과 출력
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # 속도 출력
    t = tuple(x.t / seen * 1e3 for x in dt)  # 이미지당 속도
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # 플롯
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # JSON 저장
    if save_json and len(jdict):
        if weights:
            w = Path(weights[0] if isinstance(weights, list) else weights).stem
        else:
            w = f'epoch{epoch}'
        pred_json = str(save_dir / f"{w}_predictions.json")  # 예측
        LOGGER.info(f"\nSaving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f, indent=2)
        
        # KAIST Multispectral Pedestrian Dataset 평가
        try:
            # HACK: validation set에 대한 KAIST_annotation.json 생성 필요
            ann_file = 'utils/eval/KAIST_val-D_annotation.json'
            if not os.path.exists(ann_file):
                raise FileNotFoundError(f'Please generate {ann_file} for your validation set. (See utils/eval/generate_kaist_ann_json.py)')
            os.system(f"python3 utils/eval/kaisteval.py --annFile {ann_file} --rstFile {pred_json}")
        except Exception as e:
            LOGGER.info(f"kaisteval unable to run: {e}")

    # 결과 반환
    model.float()  # 학습용
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--rgbt", action="store_true", help="use RGB-T input")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # YAML 확인
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    if opt.task in ("train", "val", "test"):  # 일반 실행
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))
    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # 속도 벤치마크
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)
        elif opt.task == "study":  # 속도 vs mAP 벤치마크
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # 저장할 파일 이름
                x, y = list(range(256, 1536 + 128, 128)), []  # x축(이미지 크기), y축
                for opt.imgsz in x:  # 이미지 크기
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # 결과와 시간
                np.savetxt(f, y, fmt="%10.4g")  # 저장
                os.system("zip -r study.zip study_*.txt")
                plot_val_study(x=x)  # 플롯
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
