from ultralytics import YOLO

# 加载您刚训练好的模型
model = YOLO('runs/detect/zhuang_chris_rtx4070/weights/best.pt')

# 导出为 ONNX 格式（最通用）
model.export(format='onnx')

print("✅ 模型已导出！")