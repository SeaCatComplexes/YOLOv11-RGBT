import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/teai/gwf_file/YOLOv11-RGBT/ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-midfusion-P3.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'/home/teai/gwf_file/YOLOv11-RGBT/ultralytics/cfg/datasets/M3FD.yaml',
                cache=False,
                imgsz=640,
                epochs=30,
                batch=4,
                close_mosaic=5,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=4,
                project='M3FD',
                name='M3FD_midfusion',  # save to project/name
                )