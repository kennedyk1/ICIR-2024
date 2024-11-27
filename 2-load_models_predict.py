import os
from ultralytics import YOLO

yolov5s = {
    'model':'yolov5su',
    'rgb_weights':'YOLO_NPT_WEIGHTS/yolov5s/rgb/weights/best.pt',
    'thermal_weights':'YOLO_NPT_WEIGHTS/yolov5s/thermal/weights/best.pt'
}

yolov6s = {
    'model':'yolov6s',
    'rgb_weights':'YOLO_NPT_WEIGHTS/yolov6s/rgb/weights/best.pt',
    'thermal_weights':'YOLO_NPT_WEIGHTS/yolov6s/thermal/weights/best.pt'
}

yolov8s = {
    'model':'yolov8s',
    'rgb_weights':'YOLO_NPT_WEIGHTS/yolov8s/rgb/weights/best.pt',
    'thermal_weights':'YOLO_NPT_WEIGHTS/yolov8s/thermal/weights/best.pt'
}

yolov9s = {
    'model':'yolov9s',
    'rgb_weights':'YOLO_NPT_WEIGHTS/yolov9s/rgb/weights/best.pt',
    'thermal_weights':'YOLO_NPT_WEIGHTS/yolov9s/thermal/weights/best.pt'
}

yolov10s = {
    'model':'yolov10s',
    'rgb_weights':'YOLO_NPT_WEIGHTS/yolov10s/rgb/weights/best.pt',
    'thermal_weights':'YOLO_NPT_WEIGHTS/yolov10s/thermal/weights/best.pt'
}

yolov11s = {
    'model':'yolo11s',
    'rgb_weights':'YOLO_NPT_WEIGHTS/yolo11s/rgb/weights/best.pt',
    'thermal_weights':'YOLO_NPT_WEIGHTS/yolo11s/thermal/weights/best.pt'
}

yolo_models =  [yolov5s,yolov6s,yolov8s,yolov9s,yolov10s,yolov11s]

def get_validation(models):
    for yolo_model in models:
        print(f'============== {yolo_model['model']} ==============')
        print("RGB")
        model = YOLO(yolo_model['rgb_weights'])
        model.val(data='rgb.yaml', imgsz=640, batch=16, conf=0.25, iou=0.5, split="test", project='Validation/'+yolo_model['model'], name="rgb")
        print("Thermal")
        model = YOLO(yolo_model['thermal_weights'])
        model.val(data='thermal.yaml', imgsz=640, batch=16, conf=0.25, iou=0.5, device="0", split="test", project='Validation/'+yolo_model['model'], name="thermal")
        
def get_predictions(models):
    for yolo_model in models:
        print(f'============== {yolo_model['model']} ==============')
        print("RGB")
        model = YOLO(yolo_model['rgb_weights'])
        model.predict(source='datasets/MID-3K/test/rgb/images',imgsz=640, conf=0.25, iou=0.5, save=True, save_txt=True, save_conf=True, project='YOLO_predictions/'+yolo_model['model'], name="rgb")
        print("Thermal")
        model = YOLO(yolo_model['thermal_weights'])
        model.predict(source='datasets/MID-3K/test/thermal/images',imgsz=640, conf=0.25, iou=0.5, save=True, save_txt=True, save_conf=True, project='YOLO_predictions/'+yolo_model['model'], name="thermal")


get_validation(yolo_models) #GET VALIDATION VALUES
#get_predictions(yolo_models) #SAVE PREDICTIONS TO TEXT FILE
