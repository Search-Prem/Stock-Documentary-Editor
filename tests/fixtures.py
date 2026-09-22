"""Test-only helpers: synthesize a 'narration' with known sentence offsets, and fake stock clips."""
import subprocess, wave, struct, os, random
from pathlib import Path

SAMPLE_SCRIPT = """Have you ever noticed the leaves of your plant turning yellow? Yellow leaves do not always mean that your plant is dying.

One of the most common causes is overwatering. Overwatering can prevent oxygen from reaching the roots, and eventually cause the leaves to turn yellow.

Plants absorb water through their roots. When the soil stays wet for too long, the roots cannot breathe. Healthy roots need both moisture and air, so let the top of the soil dry out between waterings, and always check the soil with your finger before you water again, because roots that sit in soggy soil for days will slowly suffocate, turn brown and soft, and eventually stop feeding the rest of the plant."""


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.getframerate(), w.readframes(w.getnframes())


def make_narration(sentences, out_path, gaps=None, lead=0.5, tail=0.6, speed=150, seed=3):
    """Returns (duration, true_starts) where true_starts[i] = time the i-th sentence audio begins."""
    rnd = random.Random(seed)
    tmp = Path(out_path).parent / "_tts"
    tmp.mkdir(parents=True, exist_ok=True)
    rate, chunks, true_starts, t = None, [], [], 0.0

    def silence(sec, rate):
        return b"\x00\x00" * int(sec * rate)

    parts = []
    for i, s in enumerate(sentences):
        f = tmp / f"s{i}.wav"
        subprocess.run(["espeak-ng", "-v", "en-us", "-s", str(speed), "-w", str(f), s], check=True,
                       capture_output=True)
        rate, pcm = _read_wav(f)
        parts.append(pcm)
    body = b""
    body += silence(lead, rate)
    t = lead
    for i, pcm in enumerate(parts):
        true_starts.append(t)
        body += pcm
        t += len(pcm) / 2 / rate
        gap = (gaps[i] if gaps else rnd.uniform(0.55, 1.1)) if i < len(parts) - 1 else tail
        body += silence(gap, rate)
        t += gap
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(body)
    return len(body) / 2 / rate, true_starts


def make_color_clip(path, color, dur, w=1920, h=1080, fps=30, corrupt=False):
    """Solid-colour test clip (each clip a distinct colour so cuts can be detected frame by frame)."""
    from core.util import run_ffmpeg
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:r={fps}:d={dur}",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)])
    if corrupt:
        data = Path(path).read_bytes()
        Path(path).write_bytes(data[: len(data) // 3])
    return path


def frame_colors(video_path):
    """Mean RGB of every frame (1x1 downscale) -> list of (r,g,b)."""
    from core.util import find_binary
    r = subprocess.run([find_binary("ffmpeg"), "-v", "error", "-i", str(video_path), "-vf", "scale=1:1,format=rgb24",
                        "-f", "rawvideo", "-"], capture_output=True, check=True)
    b = r.stdout
    return [tuple(b[i:i + 3]) for i in range(0, len(b) - 2, 3)]
