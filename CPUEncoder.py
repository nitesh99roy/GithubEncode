#!/usr/bin/env python3
"""
CPUEncoder.py - CPU (libx264) encoder for 480p / 720p / 1080p. No watermark.

 - Put videos in Encode_Input/ (next to this script). Outputs go to Encode_Output/.
 - One decode per file, split into every selected resolution in a single ffmpeg run.
 - Smart black-bar crop detection (9 sample points, consensus vote).
 - Writes to .part files first and renames only after a successful encode, so a
   crashed/killed run never leaves a half-finished file that later gets skipped.
 - Never upscales: a resolution larger than the source is skipped.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter

# -------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------
INPUT_FOLDER = "Encode_Input"
OUTPUT_FOLDER = "Encode_Output"

# libx264 speed/quality trade-off: ultrafast, superfast, veryfast, faster, fast,
# medium, slow, slower, veryslow. Slower = better quality per MB, but much slower.
X264_PRESET = "slow"
# Content type: "film" (live action), "animation" (cartoons/anime), "grain" (grainy
# film), "" (none). Sets deblock / psy-rd / AQ strength to suit the content.
X264_TUNE = "film"

# Advanced settings for the best quality per MB (applied on top of the preset).
# Roughly "veryslow" quality: ~2x slower than plain "slow", ~5-10% smaller files.
X264_ADVANCED = True
X264_ADVANCED_PARAMS = {
    "bframes": 8,          # up to 8 B-frames in a row (B-frames are the cheapest frames)
    "b-adapt": 2,          # best method for deciding where B-frames go
    "direct": "auto",      # picks the best B-frame motion prediction per frame
    "me": "umh",           # wider motion search: finds movement more accurately
    "merange": 24,         # how far (pixels) motion search looks
    "subme": 10,           # most precise sub-pixel motion + mode decisions
    "trellis": 2,          # rate-distortion optimised quantisation on every block
    "rc-lookahead": 60,    # look 60 frames ahead when spending bits (mbtree)
    "aq-mode": 3,          # adaptive quantisation with dark-scene bias (less banding in dark areas)
    "fast-pskip": 0,       # don't skip blocks early - keeps detail in flat/slow areas
}
X264_LEVEL = "4.1"         # plays on practically every TV / phone / browser

# Quality: CRF (lower = better/bigger) with a max bitrate cap to keep file sizes in check.
# box = the frame the video is fitted into (aspect ratio is always kept).
# ref = reference frames (more = better compression; 1080p kept at 4 for level 4.1).
# audio_channels: downmix to this many channels if the source has more (None = keep).
FORMATS = {
    "480p": {"box": (720, 480), "crf": 23, "maxrate": "700k", "bufsize": "1400k", "ref": 8,
             "audio_bitrate": "96k", "audio_channels": 2},
    "720p": {"box": (1280, 720), "crf": 22, "maxrate": "1500k", "bufsize": "3000k", "ref": 6,
             "audio_bitrate": "128k", "audio_channels": 2},
    "1080p": {"box": (1920, 1080), "crf": 21, "maxrate": "3000k", "bufsize": "6000k", "ref": 4,
              "audio_bitrate": "256k", "audio_channels": None},
}


def x264_args(tag):
    """libx264 video arguments for one resolution (shared with split_encode.py)."""
    fmt = FORMATS[tag]
    args = ["-c:v", "libx264", "-preset", X264_PRESET, "-profile:v", "high",
            "-level:v", X264_LEVEL, "-crf", str(fmt["crf"]),
            "-maxrate", fmt["maxrate"], "-bufsize", fmt["bufsize"]]
    if X264_TUNE:
        args += ["-tune", X264_TUNE]
    params = {"ref": fmt["ref"]}
    if X264_ADVANCED:
        params.update(X264_ADVANCED_PARAMS)
    args += ["-x264-params", ":".join(f"{k}={v}" for k, v in params.items())]
    return args

# Skip a resolution when the (cropped) source is smaller than this fraction of its box
# in BOTH directions - i.e. it would only be an upscale (e.g. 1080p from a 720p source).
SKIP_UPSCALE = True
UPSCALE_TOLERANCE = 0.9

# Make the first subtitle track the default one (same behaviour as FinalEncoder.py)
DEFAULT_FIRST_SUBTITLE = True

# Kill ffmpeg if it prints nothing for this long (a normal encode prints progress constantly)
FFMPEG_STALL_TIMEOUT = 300

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv',
                    '.mpeg', '.mpg', '.m4v', '.3gp', '.webm', '.ts', '.m2ts')

# Subtitle codecs that MKV can't store as-is -> converted to SRT
SUBS_TO_SRT = {"mov_text"}


# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.2f}s"


def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def log_line(output_dir, text):
    with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
        f.write(text + "\n")


def generate_output_filename(base_name, target_quality):
    """Replace the quality tag in the source name (or insert it after the year)."""
    if base_name.endswith('_Source'):
        base_name = base_name[:-7]

    pattern = r'\b(480p|576p|720p|1080p|2160p|4k)\b'
    if re.search(pattern, base_name, re.IGNORECASE):
        return re.sub(pattern, target_quality, base_name, count=1, flags=re.IGNORECASE)

    year_pattern = r'(\b(?:19|20)\d{2}\b)'
    if re.search(year_pattern, base_name):
        return re.sub(year_pattern, rf"\1.{target_quality}", base_name, count=1)
    return f"{base_name}.{target_quality}"


def part_path(final_path):
    root, ext = os.path.splitext(final_path)
    return f"{root}.part{ext}"


def parse_ratio(text):
    try:
        num, den = (int(x) for x in str(text).split(":"))
        return num / den if num > 0 and den > 0 else 1.0
    except (ValueError, ZeroDivisionError):
        return 1.0


def probe(path):
    """One ffprobe call: main video stream, duration, audio channels, subtitles, attachments."""
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration:stream=index,codec_type,codec_name,width,height,channels,"
           "sample_aspect_ratio:stream_disposition=attached_pic",
           "-of", "json", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="ignore", timeout=60)
        data = json.loads(r.stdout or "{}")
    except Exception:
        return None
    if r.returncode != 0:
        return None

    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos or not videos[0].get("width"):
        return None
    v = videos[0]
    try:
        duration = float(data.get("format", {}).get("duration") or 0)
    except ValueError:
        duration = 0.0

    return {
        "vindex": v["index"],
        "width": int(v["width"]),
        "height": int(v["height"]),
        "sar": parse_ratio(v.get("sample_aspect_ratio", "1:1")),
        "duration": duration,
        "audio": [{"channels": int(s.get("channels") or 0)}
                  for s in streams if s.get("codec_type") == "audio"],
        "subs": [{"index": s["index"], "codec": s.get("codec_name", "")}
                 for s in streams if s.get("codec_type") == "subtitle"],
        "has_attachments": any(s.get("codec_type") == "attachment" for s in streams),
    }


# -------------------------------------------------------------------------------
# Smart crop detection
# -------------------------------------------------------------------------------
def smart_crop_detect(input_path, info):
    """Sample 9 points (10%..90%), vote on the crop. Returns a crop dict or None."""
    duration, orig_w, orig_h = info["duration"], info["width"], info["height"]
    if duration <= 0:
        print("⚠️  Unknown duration - skipping crop detection")
        return None

    print(f"🔍 Detecting black bars (9 sample points across {format_time(duration)})...")
    all_crops = []
    for pct in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-ss', f'{duration * pct:.2f}',
               '-i', input_path, '-map', f'0:{info["vindex"]}',
               # limit as a fraction (24/255) so it works for 8-bit and 10-bit sources
               '-vf', 'cropdetect=limit=0.094:round=2:reset=0',
               '-f', 'null', '-t', '3', '-']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                               errors='ignore', timeout=60)
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Sample at {pct * 100:.0f}% timed out")
            continue
        found = re.findall(r'crop=(\d+):(\d+):(\d+):(\d+)', r.stderr)
        if found:
            all_crops.append(tuple(map(int, found[-1])))

    if not all_crops:
        print("❌ No crop data collected - encoding without crop")
        return None

    best, count = Counter(all_crops).most_common(1)[0]
    confidence = count / len(all_crops) * 100

    # Low agreement -> group crops within 4px of each other
    if confidence < 50 and len(all_crops) >= 3:
        groups = {}
        for c in all_crops:
            for key in groups:
                if all(abs(c[i] - key[i]) <= 4 for i in range(4)):
                    groups[key].append(c)
                    break
            else:
                groups[c] = [c]
        largest = max(groups.values(), key=len)
        confidence = len(largest) / len(all_crops) * 100
        if confidence < 40:
            print(f"⚠️  Crop samples disagree ({confidence:.0f}% best match) - encoding without crop")
            return None
        best = max(largest, key=lambda c: c[0] * c[1])  # most conservative = biggest picture
        count = len(largest)

    width, height, x, y = best
    left, top = x, y
    right, bottom = orig_w - width - x, orig_h - height - y
    if left + right + top + bottom <= 10:
        print("✨ No significant black bars")
        return None

    width, height, x, y = width - width % 2, height - height % 2, x - x % 2, y - y % 2
    print(f"✅ Crop: {orig_w}x{orig_h} → {width}x{height} "
          f"(L={left} R={right} T={top} B={bottom}, {confidence:.0f}% agree, {count}/{len(all_crops)})")
    return {"ffmpeg_crop": f"crop={width}:{height}:{x}:{y}", "w": width, "h": height,
            "original_size": f"{orig_w}x{orig_h}", "cropped_size": f"{width}x{height}"}


# -------------------------------------------------------------------------------
# ffmpeg command building
# -------------------------------------------------------------------------------
def build_filter(info, crop, targets):
    """One decode -> (fix anamorphic) -> crop -> split -> per-output scale -> yuv420p."""
    pre = []
    if crop:
        pre.append(crop["ffmpeg_crop"])
    if abs(info["sar"] - 1.0) > 0.01:
        # Anamorphic source (e.g. DVD): convert to square pixels so the picture isn't squashed
        pre.append("scale='trunc(iw*sar/2)*2':ih")
    head = f"[0:{info['vindex']}]" + ",".join(pre)

    def out_chain(t):
        bw, bh = FORMATS[t]["box"]
        return (f"scale={bw}:{bh}:force_original_aspect_ratio=decrease:force_divisible_by=2"
                f":flags=lanczos,setsar=1,format=yuv420p[v_{t}]")

    if len(targets) == 1:
        return head + ("," if pre else "") + out_chain(targets[0])
    labels = "".join(f"[s{i}]" for i in range(len(targets)))
    chains = [head + ("," if pre else "") + f"split={len(targets)}{labels}"]
    chains += [f"[s{i}]" + out_chain(t) for i, t in enumerate(targets)]
    return ";".join(chains)


def build_output_args(info, tag, out_path):
    fmt = FORMATS[tag]
    args = ["-map", f"[v_{tag}]", "-map", "0:a?"]
    for s in info["subs"]:
        args += ["-map", f"0:{s['index']}"]
    if info["has_attachments"]:
        args += ["-map", "0:t?"]                     # fonts for ASS subtitles
    args += ["-map_metadata", "0", "-map_chapters", "0"]

    # Video
    args += x264_args(tag)

    # Audio - every track to AAC, downmixed only if it has more channels than wanted
    args += ["-c:a", "aac", "-b:a", fmt["audio_bitrate"]]
    if fmt["audio_channels"]:
        for i, a in enumerate(info["audio"]):
            if a["channels"] > fmt["audio_channels"]:
                args += [f"-ac:a:{i}", str(fmt["audio_channels"])]

    # Subtitles - copy, except formats MKV can't hold
    for i, s in enumerate(info["subs"]):
        args += [f"-c:s:{i}", "srt" if s["codec"] in SUBS_TO_SRT else "copy"]
    if info["subs"] and DEFAULT_FIRST_SUBTITLE:
        args += ["-disposition:s:0", "default"]
    if info["has_attachments"]:
        args += ["-c:t", "copy"]

    args += ["-max_muxing_queue_size", "4096", out_path]
    return args


def run_ffmpeg(cmd):
    """Run ffmpeg showing a single updating progress line. Returns (return_code, log_lines)."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    lines = []
    last_output = [time.time()]
    stalled = [False]
    done = threading.Event()

    def watchdog():
        while not done.wait(5):
            if time.time() - last_output[0] > FFMPEG_STALL_TIMEOUT:
                stalled[0] = True
                print(f"\n⏱️  No ffmpeg output for {FFMPEG_STALL_TIMEOUT}s - killing stalled process.")
                try:
                    process.kill()
                except Exception:
                    pass
                return

    threading.Thread(target=watchdog, daemon=True).start()
    progress_shown = False
    try:
        for line in process.stdout:
            last_output[0] = time.time()
            text = line.rstrip()
            if text.startswith("frame=") or text.startswith("size="):
                print("\r" + text[:150].ljust(150), end="", flush=True)
                progress_shown = True
                continue
            if progress_shown:
                print()
                progress_shown = False
            print(text)
            lines.append(line)
        rc = process.wait()
    except KeyboardInterrupt:
        done.set()
        process.terminate()
        process.wait()
        raise
    finally:
        done.set()
    if progress_shown:
        print()
    if stalled[0]:
        lines.append(f"\n[CPUEncoder] Killed: no ffmpeg output for {FFMPEG_STALL_TIMEOUT}s\n")
        rc = rc or 1
    return rc, lines


# -------------------------------------------------------------------------------
# Encode one file
# -------------------------------------------------------------------------------
def encode_file(video, input_dir, output_dir, tags, enable_crop, crop_cache=None):
    """Encode `video` into every resolution in `tags` with one ffmpeg run.
    crop_cache: dict shared across a TV series {(w, h): crop_or_None} - detection runs
    once per source resolution, and 'no black bars' is cached too."""
    print(f"\n🎬 ========== {video} ==========")
    start = time.time()
    source = os.path.join(input_dir, video)
    base = os.path.splitext(video)[0]

    info = probe(source)
    if info is None:
        print("❌ Could not read this file (ffprobe failed) - skipping")
        log_line(output_dir, f"{video}: SKIPPED - unreadable input")
        return False
    print(f"📐 Source: {info['width']}x{info['height']}, {format_time(info['duration'])}, "
          f"{len(info['audio'])} audio, {len(info['subs'])} subtitle track(s)")

    # Which outputs still need encoding
    todo = []
    for tag in tags:
        final = os.path.join(output_dir, generate_output_filename(base, tag) + ".mkv")
        if os.path.exists(final):
            print(f"⏭️  Exists, skipping: {os.path.basename(final)}")
        else:
            todo.append({"tag": tag, "final": final, "part": part_path(final)})
    if not todo:
        return True

    # Crop (cached per source resolution in series mode)
    crop = None
    if enable_crop:
        key = (info["width"], info["height"])
        if crop_cache is not None and key in crop_cache:
            crop = crop_cache[key]
            print(f"🎬 Using series crop: {crop['ffmpeg_crop'] if crop else 'none (no black bars)'}")
        else:
            crop = smart_crop_detect(source, info)
            if crop_cache is not None:
                crop_cache[key] = crop

    # Don't upscale
    eff_w = (crop["w"] if crop else info["width"]) * info["sar"]
    eff_h = crop["h"] if crop else info["height"]
    if SKIP_UPSCALE:
        kept = []
        for t in todo:
            bw, bh = FORMATS[t["tag"]]["box"]
            if eff_w < bw * UPSCALE_TOLERANCE and eff_h < bh * UPSCALE_TOLERANCE:
                print(f"⏭️  {t['tag']}: source ({eff_w:.0f}x{eff_h}) is smaller - skipped (no upscaling)")
            else:
                kept.append(t)
        todo = kept
        if not todo:
            return True

    tags_todo = [t["tag"] for t in todo]
    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-i", source,
           "-filter_complex", build_filter(info, crop, tags_todo)]
    for t in todo:
        cmd += build_output_args(info, t["tag"], t["part"])

    print(f"🚀 Encoding {' + '.join(tags_todo)} on CPU (libx264, preset {X264_PRESET})")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))

    def cleanup():
        for t in todo:
            if os.path.exists(t["part"]):
                try:
                    os.remove(t["part"])
                except OSError:
                    pass

    try:
        cleanup()
        rc, lines = run_ffmpeg(cmd)
        if rc != 0:
            log_path = os.path.join(output_dir, f"error_log_{re.sub(r'[^A-Za-z0-9]', '_', base)}.txt")
            with open(log_path, "w", encoding="utf-8") as ef:
                ef.write("FFmpeg command: " + " ".join(cmd) + "\n\nOutput:\n")
                ef.writelines(lines)
            print(f"📝 Error log: {log_path}")
            raise RuntimeError(f"ffmpeg exited with code {rc}")

        for t in todo:
            size = os.path.getsize(t["part"]) if os.path.exists(t["part"]) else 0
            if size < 1024:
                raise RuntimeError(f"output missing or too small ({size} bytes): {t['part']}")
        for t in todo:
            size = os.path.getsize(t["part"])
            os.replace(t["part"], t["final"])
            print(f"✅ {os.path.basename(t['final'])} - {size / (1024 * 1024):.2f} MB")

        elapsed = time.time() - start
        crop_note = f" (cropped {crop['original_size']} → {crop['cropped_size']})" if crop else ""
        print(f"🎉 Done '{video}' in {format_time(elapsed)}{crop_note}")
        log_line(output_dir, f"{video} - CPU x264 {'+'.join(tags_todo)}{crop_note}: {format_time(elapsed)}")
        return True

    except KeyboardInterrupt:
        cleanup()
        raise
    except Exception as e:
        cleanup()
        elapsed = time.time() - start
        print(f"❌ Failed '{video}' after {format_time(elapsed)}: {e}")
        log_line(output_dir, f"{video} - CPU x264: FAILED after {format_time(elapsed)} - {e}")
        return False


# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------
def check_tools():
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], check=True, capture_output=True, timeout=15)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print(f"❌ {tool} not found. Install FFmpeg and add it to PATH.")
            return False
    return True


def main():
    print("🎬 CPU Encoder - libx264 480p / 720p / 1080p (no watermark)")
    if not check_tools():
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, INPUT_FOLDER)
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
        print(f"📁 Created {input_dir} - put your videos there and run again.")
        sys.exit(0)
    os.makedirs(output_dir, exist_ok=True)

    videos = sorted(f for f in os.listdir(input_dir) if is_video_file(f))
    if not videos:
        print(f"❌ No video files in {input_dir}")
        sys.exit(0)
    print(f"📹 Found {len(videos)} video file(s)\n")

    try:
        enable_crop = input("Remove black bars automatically? (y/n, default y): ").strip().lower() != "n"
    except EOFError:
        enable_crop = True

    print("\n📋 Select option:")
    print("1) 480p only")
    print("2) 720p only")
    print("3) 1080p only")
    print("4) 480p + 720p + 1080p  (one decode, all three at once)  [default]")
    print("5) TV SERIES - 480p + 720p + 1080p, black bars detected once and reused for every episode")
    options = {1: ["480p"], 2: ["720p"], 3: ["1080p"],
               4: ["480p", "720p", "1080p"], 5: ["480p", "720p", "1080p"]}
    try:
        option = int(input("Enter option (1-5): ").strip())
        if option not in options:
            raise ValueError
    except (ValueError, EOFError):
        print("Using default option 4.")
        option = 4

    tags = options[option]
    crop_cache = {} if option == 5 else None
    print(f"\n✅ Formats: {' + '.join(tags)} | crop: {'on' if enable_crop else 'off'} | "
          f"preset: {X264_PRESET}")

    overall = time.time()
    ok = failed = 0
    try:
        for i, video in enumerate(videos, 1):
            print(f"\n📦 File {i}/{len(videos)}")
            if encode_file(video, input_dir, output_dir, tags, enable_crop, crop_cache):
                ok += 1
            else:
                failed += 1
            if i < len(videos):
                elapsed = time.time() - overall
                eta = elapsed / i * (len(videos) - i)
                print(f"📊 Progress {i}/{len(videos)} | elapsed {format_time(elapsed)} | ETA ~{format_time(eta)}")
    except KeyboardInterrupt:
        print("\n⏸️  Stopped by user (unfinished .part files removed)")
        sys.exit(0)

    total = time.time() - overall
    print("\n🎉 ========== ALL DONE ==========")
    print(f"⏱️  Total time: {format_time(total)}")
    print(f"✅ Succeeded: {ok}   ❌ Failed: {failed}")
    log_line(output_dir, f"\nOVERALL - CPU option {option}: {format_time(total)} - Success: {ok}, Failed: {failed}")


if __name__ == "__main__":
    main()
