import cv2
import numpy as np
import onnxruntime as ort
import time

def letterbox(img, size=640):
    """保持宽高比缩放到 size×size，灰边填充。返回(处理后图, 缩放比, (左边距, 上边距))"""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)                    # 取较小的缩放比，保证不超出
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    left, top = int(round(pad_w - 0.1)), int(round(pad_h - 0.1))
    right, bottom = size - new_w - left, size - new_h - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, ratio, (left, top)

# 1. 前处理：letterbox 缩放填充
img = cv2.imread("../images/zidane.jpg")
t0 = time.perf_counter()                                   # 前处理计时起点
padded, ratio, (left, top) = letterbox(img, 640)

# 2. BGR 转 RGB、HWC 转 CHW、加 batch 维、归一化
blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
t1 = time.perf_counter()                                   # 前处理结束 / 推理开始

# 3. 推理（打印输入输出名，确认输入名是 "images"、输出名是 "output0"）
sess = ort.InferenceSession("../models/yolov8n.onnx")
print(sess.get_inputs()[0].name, sess.get_outputs()[0].name)
out = sess.run(None, {"images": blob})[0]    # [1, 84, 8400]
pred = out[0]                                 # [84, 8400]
t2 = time.perf_counter()                                   # 推理结束 / 后处理开始

# 4. 解析：前 4 行是框，后 80 行是类别得分
boxes_xywh = pred[:4].T                       # [8400, 4]
scores = pred[4:].T                           # [8400, 80]

CONF = 0.25                                   # 置信度阈值
keep = []
for i in range(8400):
    cls = int(scores[i].argmax())
    conf = float(scores[i].max())
    if conf < CONF:
        continue
    cx, cy, w, h = boxes_xywh[i]
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    keep.append([x1, y1, x2, y2, conf, cls])

# 5. NMS（自己写，别用 cv2.dnn.NMSBoxes——C++ 端也要手写，两边算法一致才好对齐）
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

keep.sort(key=lambda x: -x[4])                # 按置信度降序
result = []
while keep:
    result.append(keep[0])
    keep = [b for b in keep[1:] if iou(keep[0], b) < 0.45]

# 6. 坐标还原 + 输出三件套（打印、画框存图、写基准文本）
with open("../output/baseline_zidane.txt", "w") as f:
    for b in result:
        x1 = (b[0] - left) / ratio
        y1 = (b[1] - top) / ratio
        x2 = (b[2] - left) / ratio
        y2 = (b[3] - top) / ratio
        # 打印：类别号、置信度、还原后的框
        print(f"class {b[5]}  conf {b[4]:.3f}  box {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
        # 画框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 基准文本：C++ 端对齐就靠它，格式别改
        f.write(f"{b[5]} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {b[4]:.4f}\n")

cv2.imwrite("../output/zidane_result.jpg", img)

# 三段耗时打印（和 C++ 端 TimeIt 对应）
t3 = time.perf_counter()                                   # 后处理结束
print(f"preprocess: {(t1 - t0) * 1000:.2f} ms")
print(f"inference:  {(t2 - t1) * 1000:.2f} ms")
print(f"postprocess: {(t3 - t2) * 1000:.2f} ms")