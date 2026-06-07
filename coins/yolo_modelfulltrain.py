yolo_model.train(
    data=os.path.join(base_dir, "dataset.yml"),
    epochs=3,
    imgsz=256,
    batch=2,
    device="cpu",
    amp=True,
    verbose=True,
)
