from ultralytics import YOLOv10

model = YOLOv10(r'/Users/zhouxue/Desktop/yolov10-main-bus/runs/detect/train_v10/weights/best.pt')

model.predict(source=r'/Users/zhouxue/Desktop/yolov10-main-bus/datasets/Data/test/images',save = True)



