import os
from time import sleep

# USING MAP50 AS WEIGHT
weight_rgb = 0.675 #YOLOv8s mAP50 rgb
weight_thermal = 0.739 #YOLOv8s mAP50 thermal

# THRESHOLDS
iou_th = 0.5 #INTERSECTION OVER UNION
conf_th = 0.4 #CONFIDENCE THRESHOLD

rgb_det_folder = 'LABELS_predicted/yolov8s/rgb' #RGB DETECTIONS FOLDER
thermal_det_folder = 'LABELS_predicted/yolov8s/thermal' #THERMAL DETECTIONS FOLDER

images_folder = 'datasets/MID-3K/test/rgb/images'

rgb_det = [] #ALL RGB DETECTIONS FILES
thermal_det = [] #ALL THERMAL DETECTIONS FILES

def load_all_filenames():
    rgb = os.listdir(rgb_det_folder)
    thermal = os.listdir(thermal_det_folder)
    print("Loading RGB files...")
    for i in rgb:
        rgb_det.append(i.split('.')[0])
    print("Loading Thermal files...")
    for i in thermal:
        thermal_det.append(i.split('.')[0])
    print("All files loaded...")

def load_image_filename():
    imgs = []
    tmp = os.listdir(images_folder)
    for i in tmp:
        imgs.append(i.split('.')[0])
    return imgs

def load_weight_bboxes(scene):
    rgb_bbox = read_file(os.path.join(rgb_det_folder,scene+'.txt'))
    thermal_bbox = read_file(os.path.join(thermal_det_folder,scene+'.txt'))
    max_iou = iou_th
    pair = []
    
    used_rgb = []
    used_thermal = []
    
    number_iterations = 0
    #print('Len RGB:',len(rgb_bbox),'Len Thermal:',len(thermal_bbox))
    # WHILE TO TRY ALL RGB WITH ALL THERMAL
    while number_iterations != (len(rgb_bbox) * len(thermal_bbox)):
        max_iou = iou_th
        #print('Number iterations:',number_iterations)
        # TRY TO FIND THE BEST MATCH
        for a in rgb_bbox:
            if a not in used_rgb:
                for b in thermal_bbox:
                    if b not in used_thermal:
                        iou = get_iou(a,b)
                        if iou >= max_iou:
                            max_iou = iou

        # NOW, CHOOSE THE BEST MATCH
        for a in rgb_bbox:
            if a not in used_rgb:
                for b in thermal_bbox:
                    if b not in used_thermal:
                        if get_iou(a,b) == max_iou:
                            pair.append([a,b]) #ADD RGB AND THERMAL TO PAIR LIST
                            used_rgb.append(a) #ADD RGB TO USED LIST
                            used_thermal.append(b) #ADD THERMAL TO USED LIST
        number_iterations = number_iterations + 1
    # CHOOSE REMAINNING BBOXES
    for i in rgb_bbox:
        if i not in used_rgb:
            pair.append([i,{}]) #ADD ONLY RGB TO PAIR
            used_rgb.append(i) #ADD RGB TO USED LIST
    # CHOOSE REMAINNING BBOXES
    for i in thermal_bbox:
        if i not in used_thermal:
            pair.append([{},i]) #ADD ONLY RGB TO PAIR
            used_thermal.append(i) #ADD RGB TO USED LIST

    # INSERT SOMETHING IF PAIR LIST IS EMPTY
    if len(pair) == 0:
        pair.append([{},{}])

    data = []
    for i in pair:
        wm = {}
        if i[0] != {} and i[1] != {}:
            wm = {
                'cls' : i[0]['cls'],
                'xc' : calc_weighted_mean(i[0]['xc'],i[1]['xc']),
                'yc' : calc_weighted_mean(i[0]['yc'],i[1]['yc']),
                'w' : calc_weighted_mean(i[0]['w'],i[1]['w']),
                'h' : calc_weighted_mean(i[0]['h'],i[1]['h']),
                'score': calc_weighted_mean(i[0]['score'],i[1]['score'])
                }
        
        if i[0] == {} and i[1] != {}:
            wm = {
                'cls' : i[1]['cls'],
                'xc' : i[1]['xc'],
                'yc' : i[1]['yc'],
                'w' : i[1]['w'],
                'h' : i[1]['h'],
                'score': calc_weighted_mean(0,i[1]['score'])
                }

        if i[1] == {} and i[0] != {}:
            wm = {
                'cls' : i[0]['cls'],
                'xc' : i[0]['xc'],
                'yc' : i[0]['yc'],
                'w' : i[0]['w'],
                'h' : i[0]['h'],
                'score': calc_weighted_mean(i[0]['score'],0)
                }
        
        data.append({
            'scene':scene,
            'rgb':i[0],
            'thermal':i[1],
            'wm':wm
        })
    del pair #DELETE PAIR VECTOR
    return data

def calc_weighted_mean(RGB_value,Thermal_value):
    return round(((RGB_value * weight_rgb) + (Thermal_value * weight_thermal))/(weight_rgb + weight_thermal),6)

def get_dim_BB(A):
    A = A.replace('\n','').split(' ')
    if len(A) == 6:
        BB = {
            'cls' : int(A[0]),
            'score': round(float(A[1]),6),
            'xc' : round(float(A[2]),6),
            'yc' : round(float(A[3]),6),
            'w' : round(float(A[4]),6),
            'h' : round(float(A[5]),6),
            'P1' : {'x':round((float(A[2]) - float(A[4])/2.0),6),'y':round((float(A[3]) - float(A[5])/2.0),6)},
            'P2' : {'x':round((float(A[2]) + float(A[4])/2.0),6),'y':round((float(A[3]) + float(A[5])/2.0),6)}
        }
    else:
        BB = {
            'cls' : int(A[0]),
            'xc' : round(float(A[1]),6),
            'yc' : round(float(A[2]),6),
            'w' : round(float(A[3]),6),
            'h' : round(float(A[4]),6),
            'P1' : {'x':round((float(A[1]) - float(A[3])/2.0),6),'y':round((float(A[2]) - float(A[4])/2.0),6)},
            'P2' : {'x':round((float(A[1]) + float(A[3])/2.0),6),'y':round((float(A[2]) + float(A[4])/2.0),6)}
        }
    return BB

def get_iou(boxA, boxB):
    # if boxes do not intersect
    if have_intersection(boxA, boxB) is False:
        return 0
    interArea = get_intersection_area(boxA, boxB)
    union = get_union_areas(boxA, boxB)
    # intersection over union
    iou = interArea / union
    assert iou >= 0
    return iou

def have_intersection(boxA, boxB):
    if boxA['P1']['x'] > boxB['P2']['x']:
        return False  # boxA is right of boxB
    if boxB['P1']['x'] > boxA['P2']['x']:
        return False  # boxA is left of boxB
    if boxA['P2']['y'] < boxB['P1']['y']:
        return False  # boxA is above boxB
    if boxA['P1']['y'] > boxB['P2']['y']:
        return False  # boxA is below boxB
    return True

def get_intersection_area(boxA, boxB):
    xA = max(boxA['P1']['x'], boxB['P1']['x'])
    yA = max(boxA['P1']['y'], boxB['P1']['y'])
    xB = min(boxA['P2']['x'], boxB['P2']['x'])
    yB = min(boxA['P2']['y'], boxB['P2']['y'])
    # intersection area
    return (xB - xA) * (yB - yA)

def get_union_areas(boxA, boxB):
    area_A = boxA['w'] * boxA['h']
    area_B = boxB['w'] * boxB['h']
    interArea = get_intersection_area(boxA, boxB)
    return float(area_A + area_B - interArea)

def read_file(path):
    BB = []
    tmp = []
    try:
        with open(path,'r') as f:
            tmp = f.readlines()
        for i in tmp:
            BB.append(get_dim_BB(i))
    except:
        pass
    return BB

def export_csv(data,filename):
    if '.csv' not in filename:
        filename = filename.split('.')[0] + '.csv'
    with open(filename,'w') as f:
        f.write('scene;RGB Weight;Thermal Weight;class;R_xc;R_yc;R_w;R_h;R_Score;T_xc;T_yc;T_w;T_h;T_Score;W_xc;W_yc;W_w;W_h;W_Score\n')
    with open(filename,'a') as f:
        for i in data:
            rgb = i['rgb']
            thermal = i['thermal']
            wm = i['wm']
            
            null_values = {
                'cls':'',
                'xc':'',
                'yc':'',
                'w':'',
                'h':'',
                'score':''
            }
            
            if rgb == {}:
                rgb = null_values
            if thermal == {}:
                thermal = null_values
            if wm == {}:
                wm = null_values
            
            f.write(f'{i['scene']};{weight_rgb};{weight_thermal};0;{rgb['xc']};{rgb['yc']};{rgb['w']};{rgb['h']};{rgb['score']};{thermal['xc']};{thermal['yc']};{thermal['w']};{thermal['h']};{thermal['score']};{wm['xc']};{wm['yc']};{wm['w']};{wm['h']};{wm['score']}\n')

def export_labels(data,folder='predictions-3K'):
    subfolders = ['rgb','thermal','weighted_mean']

    for subfolder in subfolders:
        try:
            os.makedirs(os.path.join(folder,subfolder))
        except:
            pass
    
    for i in data:
        with open(os.path.join(folder,'rgb',i['scene']+'.txt'),'a') as f:
            if i['rgb'] != {}:
                f.write(f'{i['rgb']['cls']} {i['rgb']['score']} {i['rgb']['xc']} {i['rgb']['yc']} {i['rgb']['w']} {i['rgb']['h']}\n')
                
        with open(os.path.join(folder,'thermal',i['scene']+'.txt'),'a') as f:
            if i['thermal'] != {}:
                f.write(f'{i['thermal']['cls']} {i['thermal']['score']} {i['thermal']['xc']} {i['thermal']['yc']} {i['thermal']['w']} {i['thermal']['h']}\n')
            
        with open(os.path.join(folder,'weighted_mean',i['scene']+'.txt'),'a') as f:
            if i['wm'] != {}:
                f.write(f'{i['wm']['cls']} {i['wm']['score']} {i['wm']['xc']} {i['wm']['yc']} {i['wm']['w']} {i['wm']['h']}\n')


if __name__ == '__main__':
    load_all_filenames()
    images = load_image_filename()
    data = []
    for image in images:
        #print('Image:',image)
        BBoxes = load_weight_bboxes(image)
        for i in BBoxes:
            data.append(i)
    export_csv(data,'late-fusion.csv')
    export_labels(data,'Late-Fusion')
