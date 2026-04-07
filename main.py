import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
import argparse
from pathlib import Path
from configs.train_config import YOLO_BEST_WEIGHTS, CNN_BEST_WEIGHTS, OUTPUTS_DIR, DEVICE, YOLO_IMGSZ
from src.simple_pipeline import Pipeline
# Change this line in main.py:
from src.simple_pipeline import SimpleHelmetPipeline

# And inside your main() function:
def main():
    parser = argparse.ArgumentParser(description="Run Helmet Violation Detection pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--output", type=str, default=None, help="Path to output")
    parser.add_argument("--yolo-weights", type=str, default=str(YOLO_BEST_WEIGHTS))
    parser.add_argument("--cnn-weights", type=str, default=str(CNN_BEST_WEIGHTS))
    parser.add_argument("--device", type=str, default=DEVICE)
    # ADDED THIS: Allows you to increase resolution for small plates
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help="Inference size")

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"ERROR: Input source not found: {source_path}")
        return

    output_path = args.output
    if output_path is None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / f"{source_path.stem}_result{source_path.suffix}"
    else:
        output_path = Path(output_path)

    # Initialize Pipeline
    pipeline = SimpleHelmetPipeline(  # Use the correct class name here
    yolo_weights=args.yolo_weights,
    classifier_weights=args.cnn_weights,
    device=args.device
    )
    
    # Pass the custom imgsz to the detector inside the pipeline
    pipeline.detector.imgsz = args.imgsz

    ext = source_path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        pipeline.process_image(source_path, output_path)
    else:
        pipeline.process_video(source_path, output_path)

if __name__ == "__main__":
    main()