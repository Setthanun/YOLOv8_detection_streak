# YOLOv8_detection_streak
เก็บรวบรวมโค้ดของ yolov8n พร้อมคำอธิบาย 

บทความที่วิเคราะห์และรีวิวสถาปัตยกรรมของ YOLOv8 - [A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS](https://arxiv.org/abs/2304.00501?utm_source=chatgpt.com)

Ultralytics YOLOv8 Docs - [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)

[![arXiv](https://img.shields.io/badge/arXiv-2304.00501-red)](https://arxiv.org/abs/2304.00501)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics%20YOLOv8-Docs-1E90FF)](https://docs.ultralytics.com/models/yolov8/)


# ขั้นตอนที่ 1: การติดตั้ง Dependencies

ติดตั้ง Python 3.8 - 3.10 - [Download python](https://www.python.org/downloads/)

ติดตั้ง Dependencies อื่นๆ
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install -U numpy opencv-python tqdm pandas matplotlib seaborn scipy
```

# ขั้นตอนที่ 2: Training

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

ในกรณีที่เริ่มเทรนใหม่

```python
results = model.train(data=r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>", epochs=5, imgsz=640, project=r"<ใส่ path สำหรับเก็บไฟล์โมเดล>", name="<ใส่ชื่อโฟลเดอร์สำหรับเก็บไฟล์โมเดล>")
```

ในกรณีที่เทรนต่อจากโมเดลที่มีอยู่แล้ว

```python
results = model.train(data=r"<ใส่ path ที่มีไฟล์ dataset.yaml อยู่>", epochs=5, imgsz=640, project=r"<ใส่ path สำหรับเก็บไฟล์โมเดล>", name="<ใส่ชื่อโฟลเดอร์สำหรับเก็บไฟล์โมเดล>", weights=r"<ใส่ path ที่เก็บไฟล์โมเดลที่เคยเทรนไว้แล้ว.pt>")
```

# ขั้นตอนที่ 3: Test

```python
model = YOLO(r"<ใส่ path ที่เก็บไฟล์โมเดลที่เทรนไว้แล้ว.pt>")
results = model.predict(source=r"<ใส่ path โฟลเดอร์ที่เก็บภาพสำหรับทดสอบ>", save=True, project=r"<ใส่ path ที่จะเก็บผลลัพธ์การทดสอบ>", name="<ใส่ชื่อโฟลเดอร์ที่จะเก็บผลลัพธ์การทดสอบ>")
```

