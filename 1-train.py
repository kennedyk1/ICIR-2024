from ultralytics import YOLO

datasets = ['rgb.yaml','thermal.yaml']
#yolo_models = ['yolov5s.pt','yolov6s.pt','yolov8s.pt','yolov9s.pt','yolov10s.pt','yolo11s.pt']
yolo_models = ['yolov6s.yaml','yolov5s.yaml','yolov8s.yaml','yolov9s.yaml','yolov10s.yaml','yolo11s.yaml']
epochs = 50


#model = YOLO('yolov6s')

#"""
for dataset in datasets:
    for yolo_model in yolo_models:
        model = YOLO(yolo_model,task='detect') # Create a new model to train
        model.train(
            epochs = epochs, #NUMBER EPOCHS TO TRAIN
            batch = 0.9, #TO USE 90% GPU MEMORY
            single_cls = True, #TO TRAIN WITH ONLY ONE CLASS
            imgsz = 640, # TO REZISE IMAGES, DEFAULT 640
            data = dataset, #TO DEFINE YAML DATASET FILE
            patience = 0, #TO AVOID EARLY STOP
            save = True, #TO SAVE CHECKPOINTS AND FINAL MODEL WEIGHTS
            project = yolo_model.split('.')[0], #NAME OF PROJECT
            name = dataset.split('.')[0], #SUB-NAME OF PROJECT or MODALITY
            plots = True #TO SHOW PLOTS OF TRAINNING AND VALIDATION METRICS
        )
#"""