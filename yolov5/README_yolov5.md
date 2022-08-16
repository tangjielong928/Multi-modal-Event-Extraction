##### 数据处理

---

- 设置数据相关配置文件：data/ccks2022.yaml
  - path / train / val 路径
  - nc类别数
  - names类别名



##### Train

---

```shell
cd ROOTPATH/yolov5

python train.py --data data/ccks2022.yaml --weights /data/zxwang/models/yolov5s/yolov5l.pt --img 640 --epochs 100 --batch-size 16
```



##### Predict

---

```shell
cd ROOTPATH/yolov5

python predict.py --input_img_source /data/zxwang/ccks2022/val/dev_images/ --weights /data/zxwang/models/yolov5s/yolov5l.pt --img 640 --epochs 100 --batch-size 16
```

