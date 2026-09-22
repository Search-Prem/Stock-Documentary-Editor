# Stock Documentary Editor

A local Python app that turns **your script + your existing narration audio** into a stock-footage documentary
video, timed to the narration. It does **not** generate narration, images, or video, and does not touch YouTube.

```
script.txt + narration.wav  ->  sentence timestamps  ->  Pexels search per sentence  ->  timeline
                            ->  review/replace in the browser  ->  FFmpeg  ->  MP4 (+ SRT)
```

## Install (Windows)

1. Install **Python 3.10+** (tick "Add to PATH") and **FFmpeg**: `winget install Gyan.FFmpeg`
   (or download a build and set `FFMPEG_PATH` in `.env`). Check with `ffmpeg -version`.
2. Double-click **setup.bat** (creates a virtual environment, installs packages, creates `.env`).
3. Open `.env` and paste your key: `PEXELS_API_KEY=...` (free at pexels.com/api).
   `PIXABAY_API_KEY` and `UNSPLASH_ACCESS_KEY` are optional fallbacks. The app works with Pexels alone.
4. Double-click **run.bat**. Your browser opens at http://127.0.0.1:5000 (only reachable from your PC).

The first time Whisper runs it downloads a speech model (`small.en` ~ 460 MB; choose `base.en` in Settings for
a smaller/faster one). It runs on CPU and needs roughly 1 GB RAM. Without Whisper installed the app still works
using silence-based timing.

## Using it

1. **Create a project**, paste the script, import the narration (WAV/MP3).
2. **Align narration to script**: every sentence gets an exact start/end (see below).
3. **Find footage**: 1-2 Pexels searches per sentence; long sentences get several clips.
4. Review each scene. **Preview** plays just that scene with its narration. **Search again** gives different
   footage (previous picks are excluded and the next queries are used). **Replace** lets you pick another stock
   clip, paste a path to your own video/image, or upload one. You can edit the search words for a scene too.
5. **Render preview**, then **Export** either *video + narration* or *video only*. Output goes to
   `projects/<name>/final/` with `subtitles.srt` and `credits.txt` (source pages of every stock clip used).

Command line alternative: `python cli.py run my_doc --script script.txt --narration narration.wav`

Projects live in `projects/<name>/` (`script/ narration/ timeline/ media/ preview/ final/`). Close the app any time;
reopening resumes where you left off. Downloads and search results are cached in `cache/` and never fetched twice.

## How the timing works

* **Whisper (best):** word timestamps are matched to *your script's* words (so a mis-heard word doesn't matter),
  then snapped to the nearest real pause. If fewer than 60% of words match, the result is rejected with a warning.
* **Silence-based fallback:** FFmpeg finds pauses; long pauses are treated as sentence gaps, short ones as commas.
* **Estimate:** word-count based, only if audio analysis fails or no narration exists. Always flagged in the UI.

Scenes tile the narration with no gaps or overlaps; the first starts at 0 and the last ends at the audio length.
Every cut is snapped to the 30 fps frame grid, so a scene boundary is within half a frame (16.7 ms) of the
timeline value and the video length matches the narration to within one frame.

## Things worth knowing

* **Pexels rate limit:** the default key allows about 200 requests/hour. The app searches only until it has enough
  footage, caches every result, and if the limit is hit it stops cleanly, keeps everything finished, and you press
  *Find footage* again later. Roughly 60 scenes fit in an hour; larger scripts finish across two sessions.
* Footage is downloaded at 1080p (not 4K) to save disk and RAM. Rendering is one scene at a time.
* Scenes marked **NEEDS ATTENTION** (no footage, or not enough distinct footage so a clip is looped) render as a
  grey placeholder if you export anyway, so one bad scene never blocks the whole video.
* Relevance is judged from Pexels' search rank and the words in each clip's page URL. It is a good first pass,
  not a mind-reader: expect to use Replace / Search again on some scenes.

## What was verified, and what was not

Tested (51 automated tests, `python -m pytest tests`) on Linux with real FFmpeg:
alignment vs known ground truth (silence tier: within ~60 ms), gap/overlap validation, frame-exact rendering
(every clip's colour checked frame by frame in the final video), crossfades, multi-clip filling, SRT, resume with
zero re-downloads, Search Again, Replace (stock/local/upload), corrupt/portrait/short/low-res clip rejection,
missing-file and rate-limit handling, and the web API and security checks.

**Not verified (please check on your machine):**
* **Live Pexels API.** It was unreachable from my test environment, so the provider was tested against a mock
  server that follows Pexels' documented response format. Search *quality* on real footage is untested.
* **Real Whisper transcription.** The model could not be downloaded. The matching logic is tested with synthetic
  and deliberately bad speech recognition, and the fallback chain works, but Whisper's accuracy on your voice
  files is unmeasured.
* **Pixabay and Unsplash providers**: written to their documented APIs but never run.
* **Windows, your RTX 2050 / 12 GB machine, and render speed.** Nothing here uses the GPU (CPU x264).
* **The browser UI was not clicked through in a real browser** (JavaScript syntax-checked; the API behind it is tested).
* **Audio-only checks:** the muxed audio length matches the narration; I did not listen to a real music mix.

Run `python -m pytest tests -q` after setup to confirm your FFmpeg build behaves the same.
