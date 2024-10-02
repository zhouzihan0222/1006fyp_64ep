import numpy as np
import pandas as pd
from ultralytics import YOLOv10
import glob, os
from PIL import Image

results = list()
metric_values = dict()

model = YOLOv10(r'./runs/detect/fold_1_v10/weights/best.pt')
# 执行验证/推理，确保提供验证集路径
result = model.predict(source='./datasets/Data/test/images')
print(result)


results.append(result)

model = YOLOv10(r'./runs/detect/fold_2_v10/weights/best.pt')
result = model.predict(source='./datasets/Data/test/images')
results.append(result)

model = YOLOv10(r'./runs/detect/fold_3_v10/weights/best.pt')
result = model.predict(source='./datasets/Data/test/images')
results.append(result)

model = YOLOv10(r'./runs/detect/fold_4_v10/weights/best.pt')
result = model.predict(source='./datasets/Data/test/images')
results.append(result)

model = YOLOv10(r'./runs/detect/fold_5_v10/weights/best.pt')
result = model.predict(source='./datasets/Data/test/images')
results.append(result)


metric_values = dict()

for result in results:
    for metric, metric_val in result.results_dict.items():
        if metric not in metric_values:
            metric_values[metric] = []
        metric_values[metric].append(metric_val)

metric_df = pd.DataFrame.from_dict(metric_values)
visualize_metric = ['mean', 'std', 'min', 'max']

metric = metric_df.describe().loc[visualize_metric]
print(metric)

metric.to_csv('./runs/detect/5predict_metrics_summary.csv', index=True)
print("Metrics saved to metrics_summary.csv")

