import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('/home/teai/gwf_file/YOLOv11-RGBT/ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-P3.yaml')
    model = YOLO('/home/teai/gwf_file/YOLOv11-RGBT/runs/M3FD/M3FD-yolo11n-RGBT-midfusion-3/weights/best.pt')
    # model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'/home/teai/gwf_file/YOLOv11-RGBT/ultralytics/cfg/datasets/M3FD.yaml',
                cache=False,
                imgsz=640,
                epochs=30,
                batch=16,
                close_mosaic=0,
                workers=2,
                device= "0",
                optimizer='SGD',  # using SGD
                # amp=False, # close amp
                # fraction=0.2,
                # pairs_rgb_ir=['visible','infrared'] , # default: ['visible','infrared'] , others: ['rgb', 'ir'],  ['images', 'images_ir'], ['images', 'image']
                use_simotm="RGBT",
                channels=4,
                project='runs/M3FD',
                name='M3FD-yolo11n-RGBT-midfusion-',
                )