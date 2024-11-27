import os
from time import sleep

yolo_pred = 'YOLO_predictions'
save_folder = 'LABELS_predicted'

models = os.listdir(yolo_pred)
modalities = ['rgb','thermal']

def convert_label(folder):
    _, model, modality, _ = folder.split('/')
    try:
        os.makedirs(os.path.join(save_folder,model,modality)) #TRY TO CREATE FOLDER, IGNORE IF EXISTS
    except:
        pass
    
    files = os.listdir(folder)
    raw_data = []

    for file in files:
        with open(os.path.join(folder,file),'r') as f: #TO READ ALL CONTENT TXT FILE
            raw_data = f.readlines()
        
        print(f'Writing {os.path.join(save_folder,model,modality,file)}')
        with open(os.path.join(save_folder,model,modality,file),'a') as f: #WRITE A NEW TXT FILE
            for i in raw_data:
                cls, xc, yc, w, h, conf = i.replace('\n','').split(' ')
                f.write(f'{cls} {conf} {xc} {yc} {w} {h}\n')

if __name__ == '__main__':
    for model in models:
        for modality in modalities:
            folder = os.path.join(yolo_pred,model,modality,'labels')
            convert_label(folder)