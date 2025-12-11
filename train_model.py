from ultralytics import YOLO

def main():
    model = YOLO('yolo11n.pt') 

    print("Starting Robust Training")
    results = model.train(
        data="path/to/multi_insect.yaml",
        epochs=100,
        patience=20,
        imgsz=960,
        batch=64,
        name='mosquito_robust',
        single_cls=True,
        device=0,
        workers=4,
        optimizer='auto', 
        lr0=0.001,
        
        erasing=0.2,
        scale=0.7,
        degrees=45.0,
        flipud=0.5,
    
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
    )

    print("Training Complete")

if __name__ == '__main__':
    main()