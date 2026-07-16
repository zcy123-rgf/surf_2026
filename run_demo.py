import argparse

from surf_bev.detectors import build_detector
from surf_bev.pipeline import run_multi_frame, run_single_frame


def parse_frame_ids(text):
    return [int(item) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Lane detection + BEV fusion demo.")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--detector", choices=["clrnet", "hough"], default="clrnet")
    parser.add_argument("--image", default="data/000001_original.jpg")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--image-pattern", default="um_{frame_id:06d}.png")
    parser.add_argument("--calib", default=None)
    parser.add_argument("--poses", default=None)
    parser.add_argument("--frame-ids", default="0,1,2")
    parser.add_argument("--ref-id", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/surf_bev")
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--clrnet-root", default="CLRNet")
    parser.add_argument("--clrnet-config", default="configs/clrnet/clr_resnet18_culane.py")
    parser.add_argument("--clrnet-checkpoint", default="weights/culane_r18.pth")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    detector_kwargs = {}
    if args.detector == "clrnet":
        detector_kwargs = {
            "clrnet_root": args.clrnet_root,
            "config": args.clrnet_config,
            "checkpoint": args.clrnet_checkpoint,
            "device": args.device,
        }
    detector = build_detector(args.detector, **detector_kwargs)

    if args.mode == "single":
        result = run_single_frame(
            detector,
            args.image,
            args.output_dir,
            calib_path=args.calib,
            camera_height=args.camera_height,
            pitch_deg=args.pitch_deg,
        )
        print(f"Detected lanes: {len(result['lanes'])}")
        print(f"Image lanes: {result['image_out']}")
        print(f"BEV result: {result['bev_out']}")
        return

    if not args.image_dir or not args.calib or not args.poses:
        raise SystemExit("--mode multi requires --image-dir, --calib and --poses")

    result = run_multi_frame(
        detector,
        args.image_dir,
        args.output_dir,
        args.calib,
        args.poses,
        parse_frame_ids(args.frame_ids),
        ref_id=args.ref_id,
        image_pattern=args.image_pattern,
        camera_height=args.camera_height,
        pitch_deg=args.pitch_deg,
    )
    print(f"Fused BEV: {result['fused_out']}")
    print(f"Debug frames: {result['debug_count']}")


if __name__ == "__main__":
    main()

