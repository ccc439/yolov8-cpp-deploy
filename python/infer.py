"""Reference implementation. Timers exclude session loading, warmup and output I/O."""
import argparse
import math
from pathlib import Path
import time
import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image, width, height):
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    # Match C++ round for positive values (Python round uses ties-to-even).
    nw = min(width, max(1, math.floor(w * ratio + 0.5)))
    nh = min(height, max(1, math.floor(h * ratio + 0.5)))
    left, top = (width - nw) // 2, (height - nh) // 2
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(resized, top, height - nh - top,
                              left, width - nw - left, cv2.BORDER_CONSTANT,
                              value=(114, 114, 114)), ratio, left, top


def iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 and math.isfinite(denom) else 0.


def postprocess(output, conf, threshold, agnostic, ratio, left, top, width, height):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] <= 4:
        raise ValueError("Expected raw YOLOv8 output [1,4+classes,candidates]")
    pred = output[0]
    boxes = []
    for i in range(pred.shape[1]):
        scores = pred[4:, i]
        scores = np.where(np.isnan(scores), -1, scores)
        cls = int(scores.argmax())
        score = float(scores[cls])
        cx, cy, w, h = map(float, pred[:4, i])
        if not math.isfinite(score) or score < conf or not all(map(math.isfinite, (cx, cy, w, h))) or w <= 0 or h <= 0:
            continue
        boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2, score, cls])
    boxes.sort(key=lambda b: -b[4])
    suppressed = [False] * len(boxes)
    result = []
    for i, a in enumerate(boxes):
        if suppressed[i]:
            continue
        result.append(a.copy())
        for j in range(i+1, len(boxes)):
            if not suppressed[j] and (agnostic or a[5] == boxes[j][5]) and iou(a, boxes[j]) >= threshold:
                suppressed[j] = True
    restored = []
    for b in result:
        b[0] = min(width, max(0., (b[0]-left)/ratio))
        b[2] = min(width, max(0., (b[2]-left)/ratio))
        b[1] = min(height, max(0., (b[1]-top)/ratio))
        b[3] = min(height, max(0., (b[3]-top)/ratio))
        if b[2] > b[0] and b[3] > b[1]:
            restored.append(b)
    return restored


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=root/"images/zidane.jpg")
    parser.add_argument("--model", type=Path, default=root/"models/yolov8n.onnx")
    parser.add_argument("--output", type=Path, default=root/"output")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.45)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--benchmark", type=int, default=1)
    parser.add_argument("--agnostic", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1) or args.size <= 0 or args.threads < 0 or args.warmup < 0 or args.benchmark < 1:
        parser.error("Invalid threshold, size, thread count or iterations")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.model), sess_options=opts, providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    if inp.type != "tensor(float)" or len(inp.shape) != 4 or (isinstance(inp.shape[0], int) and inp.shape[0] != 1) or inp.shape[1] != 3:
        parser.error("Expected float32 NCHW batch=1 input")
    height = inp.shape[2] if isinstance(inp.shape[2], int) and inp.shape[2] > 0 else args.size
    width = inp.shape[3] if isinstance(inp.shape[3], int) and inp.shape[3] > 0 else args.size

    def detect():
        start = time.perf_counter()
        padded, ratio, left, top = letterbox(image, width, height)
        blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        blob *= np.float32(1 / 255)
        t1 = time.perf_counter()
        output = session.run([out.name], {inp.name: blob})[0]
        t2 = time.perf_counter()
        boxes = postprocess(output, args.conf, args.iou, args.agnostic, ratio, left, top, image.shape[1], image.shape[0])
        end = time.perf_counter()
        return boxes, [(t1-start)*1000, (t2-t1)*1000, (end-t2)*1000, (end-start)*1000]

    for _ in range(args.warmup):
        detect()
    samples = []
    for _ in range(args.benchmark):
        boxes, times = detect()
        samples.append(times)
    mean = np.mean(samples, axis=0)
    print("preprocess / inference / postprocess / detect mean (ms):", mean)
    ordered = sorted(row[3] for row in samples)
    print("detect p50 / p95 (ms):", ordered[math.ceil(.5*len(ordered))-1], ordered[math.ceil(.95*len(ordered))-1])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/f"baseline_{args.image.stem}.txt").open("w", encoding="utf-8") as f:
        for x1, y1, x2, y2, score, cls in boxes:
            f.write(f"{cls} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.4f}\n")
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if not cv2.imwrite(str(args.output/f"{args.image.stem}_python.jpg"), image):
        raise OSError("Could not save result image")


if __name__ == "__main__":
    main()
