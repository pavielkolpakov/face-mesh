import argparse
import json
import sys
from pathlib import Path


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    if argv[0] in ("-h", "--help", "face", "body", "all"):
        return argv
    if argv[0].startswith("-"):
        return ["face", *argv]
    return ["face", "--image", argv[0], *argv[1:]]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _cmd_face(args: argparse.Namespace) -> int:
    from face_mesh.application.extract_face_features import extract_face_features
    from face_mesh.infrastructure.face.mediapipe_adapter import MediaPipeFaceLandmarkAdapter

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Missing model file: {args.model}")
        print("Provide MediaPipe face_landmarker.task via --model.")
        return 1
    image_path = Path(args.image)
    port = MediaPipeFaceLandmarkAdapter()
    payload = extract_face_features(image_path, model_path, port)
    if payload is None:
        if not image_path.exists():
            print(f"Failed to load image: {args.image}")
        else:
            print("No face detected.")
        return 1
    out = payload.as_dict()
    _write_json(Path(args.output), out)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_body(args: argparse.Namespace) -> int:
    try:
        from face_mesh.application.estimate_body_smpl import estimate_body_smpl_fused
        from face_mesh.infrastructure.body.romp_adapter import RompBodyShapeAdapter
    except ImportError as exc:
        print("Body estimation requires optional dependencies: pip install -e '.[body]'")
        print(str(exc))
        return 1

    front = Path(args.front)
    side = Path(args.side)
    if not front.is_file():
        print(f"Missing front image: {front}")
        return 1
    if not side.is_file():
        print(f"Missing side image: {side}")
        return 1

    gpu = args.gpu
    romp_ckpt = Path(args.romp_model).expanduser() if args.romp_model else None
    estimator = RompBodyShapeAdapter(gpu_device=gpu, romp_model_path=romp_ckpt)
    fused = estimate_body_smpl_fused(
        front_path=front,
        side_path=side,
        estimator=estimator,
        weight_front=args.weight_front,
        weight_side=args.weight_side,
    )
    out = fused.as_dict()
    _write_json(Path(args.output), out)
    print(json.dumps(out, indent=2))
    return 0 if fused.ok else 1


def _cmd_all(args: argparse.Namespace) -> int:
    rc = _cmd_face(
        argparse.Namespace(
            image=args.front,
            model=args.model,
            output=args.face_output,
        ),
    )
    if rc != 0:
        return rc
    return _cmd_body(
        argparse.Namespace(
            front=args.front,
            side=args.side,
            output=args.body_output,
            weight_front=args.weight_front,
            weight_side=args.weight_side,
            gpu=args.gpu,
            romp_model=args.romp_model,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face landmarks and SMPL body shape from photos.")
    sub = parser.add_subparsers(dest="command", required=False)

    p_face = sub.add_parser("face", help="MediaPipe face features (blendshapes, colors).")
    p_face.add_argument("--image", default="input.jpeg", help="Input image path.")
    p_face.add_argument("--output", default="landmarks.json", help="Output JSON path.")
    p_face.add_argument(
        "--model",
        default="face_landmarker.task",
        help="MediaPipe face_landmarker.task path.",
    )

    p_body = sub.add_parser("body", help="SMPL betas from front + side (ROMP).")
    p_body.add_argument("--front", required=True, help="Front-view body photo.")
    p_body.add_argument("--side", required=True, help="Side-view body photo.")
    p_body.add_argument("--output", default="out/body.json", help="Output JSON path.")
    p_body.add_argument("--weight-front", type=float, default=0.6, dest="weight_front")
    p_body.add_argument("--weight-side", type=float, default=0.4, dest="weight_side")
    p_body.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Torch device index; default CUDA 0 if available else CPU (-1).",
    )
    p_body.add_argument(
        "--romp-model",
        default=None,
        dest="romp_model",
        help="ROMP.pkl path; default ~/.romp/ROMP.pkl (auto-download if missing).",
    )

    p_all = sub.add_parser("all", help="Run face on front image and body on front+side.")
    p_all.add_argument("--front", required=True, help="Front photo (face + body).")
    p_all.add_argument("--side", required=True, help="Side body photo.")
    p_all.add_argument("--face-output", default="out/face.json", dest="face_output")
    p_all.add_argument("--body-output", default="out/body.json", dest="body_output")
    p_all.add_argument("--model", default="face_landmarker.task")
    p_all.add_argument("--weight-front", type=float, default=0.6, dest="weight_front")
    p_all.add_argument("--weight-side", type=float, default=0.4, dest="weight_side")
    p_all.add_argument("--gpu", type=int, default=None)
    p_all.add_argument("--romp-model", default=None, dest="romp_model")

    return parser


def main() -> None:
    argv = _normalize_argv(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        sys.exit(1)
    if command == "face":
        sys.exit(_cmd_face(args))
    if command == "body":
        sys.exit(_cmd_body(args))
    if command == "all":
        sys.exit(_cmd_all(args))
    parser.print_help()
    sys.exit(1)
