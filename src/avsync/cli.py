"""Command-line interface: `avsync check video.mp4`."""
import argparse

from avsync.sync import analyse_video


def main():
    parser = argparse.ArgumentParser(prog="avsync", description="Check audio-visual sync in a talking-head video.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="estimate the audio-lip offset of a video")
    check.add_argument("video", help="path to the video file")

    args = parser.parse_args()

    if args.command == "check":
        result = analyse_video(args.video)
        direction = "audio lags video" if result["offset_ms"] > 0 else "audio leads video"
        print(f"Verdict:    {result['verdict']}")
        print(f"Offset:     {result['offset_ms']:+.0f} ms ({result['offset_frames']:+d} frames)"
              + ("" if result["offset_ms"] == 0 else f", {direction}"))
        print(f"Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    main()