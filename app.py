"""Gradio web demo: upload a talking-head video, get an audio-visual sync verdict."""
import gradio as gr
import matplotlib
matplotlib.use("Agg")             
import matplotlib.pyplot as plt
import subprocess

from avsync.models import ensure_models
from avsync.sync import analyse_video, load_syncnet

ensure_models()
MODEL = load_syncnet()


def plot_confidence(window_confidence, fps=25):
    times = [i / fps for i in range(len(window_confidence))]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, window_confidence)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sync confidence")
    ax.set_title("Sync confidence over time")
    fig.tight_layout()
    return fig


def check(video_path):
    if video_path is None:
        return "Please upload a video.", None
    try:
        result = analyse_video(video_path, model=MODEL)
    except ValueError as e:
        return f"Couldn't analyse this video: {e}", None
    except subprocess.CalledProcessError:
        return "Couldn't read this file. Please upload a standard video (e.g. MP4).", None

    summary = (
        f"**Verdict:** {result['verdict']}\n\n"
        f"**Offset:** {result['offset_ms']:+.0f} ms ({result['offset_frames']:+d} frames)\n\n"
        f"**Confidence:** {result['confidence']:.2f}"
    )

    return summary, plot_confidence(result["window_confidence"])


demo = gr.Interface(
    fn=check,
    inputs=gr.Video(label="Talking-head video"),
    outputs=[gr.Markdown(label="Result"), gr.Plot(label="Sync over time")],
    title="AV Sync Checker",
    description="Upload a short video of someone talking. The tool estimates whether the audio and lip movement are in sync, and by how much, using a pretrained SyncNet model.",
    flagging_mode="never",
    examples=[
        ["examples/in_sync.mp4"],
        ["examples/audio_200ms_late.mp4"],
    ],
)

if __name__ == "__main__":
    demo.launch()