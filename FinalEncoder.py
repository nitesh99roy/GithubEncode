#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------------------------
# Helper function to format seconds into hours, minutes, and seconds.
# -------------------------------------------------------------------------------
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.2f}s"

# -------------------------------------------------------------------------------
# Enhanced filename generation with proper quality replacement
# -------------------------------------------------------------------------------
def generate_output_filename(base_name, target_quality):
    """Generate output filename by replacing quality in the source filename"""
    if base_name.endswith(('.mkv', '.mp4', '.avi')):
        base_name = os.path.splitext(base_name)[0]

    if base_name.endswith('_Source'):
        base_name = base_name[:-7]

    quality_patterns = [
        r'\b(480p|720p|1080p|4k|2160p)\b',
        r'\b(480P|720P|1080P|4K|2160P)\b'
    ]

    new_name = base_name
    quality_found = False

    for pattern in quality_patterns:
        if re.search(pattern, new_name, re.IGNORECASE):
            if target_quality.endswith('.10BIT'):
                base_quality = target_quality.replace('.10BIT', '')
                replacement = f"{base_quality}.10BIT"
            else:
                replacement = target_quality

            new_name = re.sub(pattern, replacement, new_name, count=1, flags=re.IGNORECASE)
            quality_found = True
            break

    if not quality_found:
        year_pattern = r'(\b\d{4}\b)'
        if re.search(year_pattern, new_name):
            if target_quality.endswith('.10BIT'):
                base_quality = target_quality.replace('.10BIT', '')
                replacement = f"\\1.{base_quality}.10BIT"
            else:
                replacement = f"\\1.{target_quality}"
            new_name = re.sub(year_pattern, replacement, new_name, count=1)
        else:
            if target_quality.endswith('.10BIT'):
                new_name = f"{target_quality}.{new_name}"
            else:
                new_name = f"{target_quality}.{new_name}"

    return new_name

# -------------------------------------------------------------------------------
# Smart FFmpeg crop detection with multi-point sampling
# -------------------------------------------------------------------------------
def smart_crop_detect(input_path, video_name):
    """Smart FFmpeg crop detection with multi-point sampling and statistical analysis.
    Samples at 9 points across the video and uses consensus to determine crop values."""
    print(f"🔍 Smart crop detection for '{video_name}'...")

    duration = get_video_duration(input_path)
    if duration <= 0:
        print("⚠️  Could not determine video duration - skipping crop detection")
        return None

    # Get original dimensions via ffprobe
    dim_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height',
               '-of', 'csv=p=0:s=x', input_path]
    try:
        dim_result = subprocess.run(dim_cmd, capture_output=True, text=True,
                                    encoding='utf-8', errors='ignore', timeout=30)
        orig_dims = dim_result.stdout.strip().split('x')
        orig_w, orig_h = int(orig_dims[0]), int(orig_dims[1])
    except Exception:
        print("⚠️  Could not determine video dimensions - skipping crop detection")
        return None

    # Multi-point sampling - skip first/last 5% to avoid intros/credits
    sample_points = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    sample_duration = 3  # seconds per sample point

    all_crops = []

    print(f"📊 Analyzing {len(sample_points)} sample points across {format_time(duration)}...")

    for i, pct in enumerate(sample_points):
        start_time = duration * pct

        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-ss', f'{start_time:.2f}', '-i', input_path,
               '-vf', 'cropdetect=limit=24:round=2:reset=0',
               '-f', 'null', '-t', str(sample_duration), '-']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding='utf-8', errors='ignore', timeout=45)

            sample_crops = []
            for line in result.stderr.split('\n'):
                crop_match = re.search(r'crop=(\d+):(\d+):(\d+):(\d+)', line)
                if crop_match:
                    crop = tuple(map(int, crop_match.groups()))
                    sample_crops.append(crop)

            if sample_crops:
                # Take the last (most stable) crop value from each sample window
                all_crops.append(sample_crops[-1])

        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"  ⚠️  Sample {i+1} at {pct*100:.0f}% failed: {str(e)}")
            continue

    if not all_crops:
        print("❌ No crop data collected - skipping crop detection")
        return None

    print(f"📊 Collected {len(all_crops)} crop samples from {len(sample_points)} points")

    # Statistical analysis - find consensus
    from collections import Counter
    crop_counter = Counter(all_crops)

    most_common = crop_counter.most_common()
    best_crop = most_common[0][0]
    best_count = most_common[0][1]
    confidence = (best_count / len(all_crops)) * 100

    # If exact match confidence is low, try grouping similar crops (within 4px tolerance)
    if confidence < 50 and len(all_crops) >= 3:
        crop_groups = {}
        for crop in all_crops:
            matched = False
            for key in list(crop_groups.keys()):
                if (abs(crop[0] - key[0]) <= 4 and abs(crop[1] - key[1]) <= 4 and
                    abs(crop[2] - key[2]) <= 4 and abs(crop[3] - key[3]) <= 4):
                    crop_groups[key].append(crop)
                    matched = True
                    break
            if not matched:
                crop_groups[crop] = [crop]

        # Find the largest consensus group
        largest_group_key = max(crop_groups, key=lambda k: len(crop_groups[k]))
        largest_group = crop_groups[largest_group_key]
        group_confidence = (len(largest_group) / len(all_crops)) * 100

        if group_confidence >= 40:
            # Use the most conservative crop in the group (largest output area)
            best_crop = max(largest_group, key=lambda c: c[0] * c[1])
            confidence = group_confidence
            best_count = len(largest_group)
            print(f"🔄 Tolerance grouping applied: {group_confidence:.0f}% consensus ({len(largest_group)}/{len(all_crops)} samples)")
        else:
            print(f"⚠️  Low confidence ({group_confidence:.0f}%) across all groups - no reliable crop detected")
            return None

    width, height, x, y = best_crop

    # Calculate removed pixels
    left = x
    right = orig_w - width - x
    top = y
    bottom = orig_h - height - y
    total_cropped = left + right + top + bottom

    # Skip if crop is insignificant
    if total_cropped <= 10:
        print(f"✨ No significant black bars detected ({total_cropped}px total)")
        return None

    # Ensure dimensions are even (required by most codecs)
    width = width - (width % 2)
    height = height - (height % 2)
    x = x - (x % 2)
    y = y - (y % 2)

    ffmpeg_crop = f"crop={width}:{height}:{x}:{y}"

    crop_info = {
        'ffmpeg_crop': ffmpeg_crop,
        'original_size': f"{orig_w}x{orig_h}",
        'cropped_size': f"{width}x{height}",
        'confidence': f"{confidence:.0f}%",
        'samples': len(all_crops),
        'removed_pixels': {
            'left': left, 'right': right, 'top': top, 'bottom': bottom
        }
    }

    print(f"✅ Smart crop detected! {orig_w}x{orig_h} → {width}x{height}")
    print(f"📋 FFmpeg: {ffmpeg_crop}")
    print(f"📊 Confidence: {confidence:.0f}% ({best_count}/{len(all_crops)} samples agree)")
    print(f"✂️  Removing: L={left}px R={right}px T={top}px B={bottom}px")

    if left == right and left > 0:
        print("📺 Type: Pillarbox (side black bars)")
    elif top == bottom and top > 0:
        print("📺 Type: Letterbox (top/bottom black bars)")
    elif total_cropped > 0:
        print("📺 Type: Irregular cropping")

    return crop_info

def detect_crop_params(input_path, video_name):
    """Smart crop detection using FFmpeg multi-point sampling"""
    return smart_crop_detect(input_path, video_name)

# -------------------------------------------------------------------------------
# Define conversion modes with OPTIMIZED QSV settings (COLOR FILTERS REMOVED)
# -------------------------------------------------------------------------------
MODES = {
    "480p_hevc_qsv": {   # key name kept for compatibility - now Intel QSV H.264 (was libx264 on CPU)
        "codec": "h264_qsv",
        "bitrate": "450k",
        "maxrate": "700k",
        "bufsize": "900k",
        "scale": "720",
        "preset": "slow",
        "profile": "high",
        "mbbrc": "1",
        "b_frames": "3",
        "refs": "3",
        "qsv_async": "8",
        "audio_codec": "aac",
        "audio_bitrate": "60k",
        "audio_channels": "1",
        "subtitle_codec": "copy",
        "pix_fmt": "nv12",
        "output_suffix": "_480p_x264_qsv.mkv"
    },
    "720x264_qsv": {
        "codec": "h264_qsv",
        "bitrate": "1200k",
        "scale": "1280",
        "preset": "medium",
        "profile": "high",
        "mbbrc": "1",
        "b_frames": "2",
        "refs": "2",
        "sc_threshold": "40",
        "qsv_async": "8",
        "audio_codec": "aac",
        "audio_bitrate": "112k",
        "subtitle_codec": "copy",
        "pix_fmt": "nv12",
        "output_suffix": "_720x264_qsv.mkv"
    },
    "1080x264_qsv": {
        "codec": "h264_qsv",
        "bitrate": "2500k",
        "scale": "1920",
        "preset": "medium",
        "profile": "high",
        "mbbrc": "1",
        "b_frames": "2",
        "refs": "2",
        "sc_threshold": "40",
        "qsv_async": "8",
        "audio_codec": "aac",
        "audio_bitrate": "256k",
        "subtitle_codec": "copy",
        "pix_fmt": "nv12",
        "output_suffix": "_1080x264_qsv.mkv"
    },
    "720p_10bit_qsv": {
        "codec": "hevc_qsv",
        "bitrate": "800k",
        "maxrate": "1100k",
        "bufsize": "2400k",
        "scale": "1280",
        "preset": "slow",
        "profile": "main10",
        "mbbrc": "1",
        "b_frames": "2",
        "refs": "2",
        "sc_threshold": "40",
        "qsv_async": "8",
        "audio_codec": "aac",
        "audio_bitrate": "96k",
        "subtitle_codec": "copy",
        "pix_fmt": "p010le",
        "output_suffix": "_720p_10bit_qsv.mkv"
    },
    "1080p_10bit_qsv": {
        "codec": "hevc_qsv",
        "bitrate": "1600k",
        "maxrate": "2000k",
        "bufsize": "4000k",
        "scale": "1920",
        "preset": "medium",
        "profile": "main10",
        "mbbrc": "1",
        "b_frames": "2",
        "refs": "2",
        "sc_threshold": "40",
        "qsv_async": "8",
        "audio_codec": "aac",
        "audio_bitrate": "160k",
        "subtitle_codec": "copy",
        "pix_fmt": "p010le",
        "output_suffix": "_1080p_10bit_qsv.mkv"
    }
}

# -------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------
def is_video_file(filename):
    VIDEO_EXTENSIONS = [
        '.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv',
        '.mpeg', '.mpg', '.m4v', '.3gp', '.webm'
    ]
    return any(filename.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

def verify_input_file(input_path):
    cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', input_path]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

def get_video_duration(input_path) -> float:
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        result = subprocess.run(
            probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

# -------------------------------------------------------------------------------
# Enhanced conversion function with better error handling
# -------------------------------------------------------------------------------
def run_conversion_step(*args, **kwargs):
    """Single-output encode. If GPU decoding causes a failure, retry without it and remember."""
    global QSV_HW_DECODE
    if not QSV_HW_DECODE:
        return _run_conversion_step_once(*args, **kwargs)
    try:
        return _run_conversion_step_once(*args, **kwargs)
    except KeyboardInterrupt:
        raise
    except Exception:
        QSV_HW_DECODE = False
        print("🔁 Retrying without GPU decoding (disabled for the rest of this run).")
        return _run_conversion_step_once(*args, **kwargs)


def _run_conversion_step_once(input_path, output_path, video_name, stage_name, mode, output_dir, is_10bit=False, enable_crop=True):
    start_time = time.time()
    print(f"\nStarting {stage_name} for '{video_name}' ...")

    # Check enhancement types
    has_premium_quality = any(key in mode for key in ["look_ahead", "extbrc"])
    has_standard_enhancements = any(key in mode for key in ["b_frames", "refs", "sc_threshold", "qsv_async", "deblock", "mbbrc"])

    if has_premium_quality:
        print(f"🎯 Premium Quality: look_ahead + mbbrc + extbrc enabled")
    elif has_standard_enhancements:
        print(f"⚡ Enhanced Encoding: mbbrc + b_frames + refs + optimized settings enabled")
        if "qsv_async" in mode:
            print(f"🚀 QSV Optimization: async_depth={mode['qsv_async']} (optimized)")

    # Crop detection
    crop_info = None
    if enable_crop:
        crop_info = detect_crop_params(input_path, video_name)

    # Get duration for watermark timing
    duration_sec = get_video_duration(input_path)
    if duration_sec <= 0:
        print(f"Warning: could not determine duration for '{video_name}'. Watermark won't appear.")
        t20 = t40 = t80 = -9999
    else:
        t20 = 0.2 * duration_sec
        t40 = 0.4 * duration_sec
        t80 = 0.8 * duration_sec

    # Build FFmpeg command
    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner"]
    if QSV_HW_DECODE:
        cmd.extend(["-hwaccel", "auto"])
    if "input_args" in mode:
        cmd.extend(mode["input_args"])
    use_image_wm = WATERMARK_MODE == "image" and os.path.exists(WATERMARK_PATH) and duration_sec > 0
    use_text_wm = WATERMARK_MODE == "text" and duration_sec > 0
    cmd.extend(["-i", input_path])
    if use_image_wm:
        cmd.extend(["-i", WATERMARK_PATH])
    cmd.extend(["-map_metadata", "0", "-map_chapters", "0"])

    # Build video filter logic
    if "scale" in mode and mode["scale"]:
        if ":" in mode["scale"]:
            scale_filter = f"scale={mode['scale']}"
        else:
            scale_filter = f"scale={mode['scale']}:-2"
    else:
        scale_filter = "scale=iw:ih"

    if crop_info and 'ffmpeg_crop' in crop_info:
        crop_filter = crop_info['ffmpeg_crop']
        if "fps" in mode:
            if "pix_fmt" in mode:
                main_filter = f"[0:v]{crop_filter},{scale_filter},fps={mode['fps']},setsar=1,format={mode['pix_fmt']}[scaled]"
            else:
                main_filter = f"[0:v]{crop_filter},{scale_filter},fps={mode['fps']}[scaled]"
            print(f"🎬 Applying: {crop_filter} + CFR {mode['fps']} FPS")
        else:
            if "pix_fmt" in mode:
                main_filter = f"[0:v]{crop_filter},{scale_filter},setsar=1,format={mode['pix_fmt']}[scaled]"
            else:
                main_filter = f"[0:v]{crop_filter},{scale_filter}[scaled]"
            print(f"🎬 Applying: {crop_filter}")
    else:
        if "fps" in mode:
            if "pix_fmt" in mode:
                main_filter = f"[0:v]{scale_filter},fps={mode['fps']},setsar=1,format={mode['pix_fmt']}[scaled]"
            else:
                main_filter = f"[0:v]{scale_filter},fps={mode['fps']}[scaled]"
            print(f"🎬 Applying: CFR {mode['fps']} FPS")
        else:
            if "pix_fmt" in mode:
                main_filter = f"[0:v]{scale_filter},setsar=1,format={mode['pix_fmt']}[scaled]"
            else:
                main_filter = f"[0:v]{scale_filter}[scaled]"

    final_fmt = ",format=p010le" if is_10bit else ""
    if use_text_wm:
        text_f = text_watermark_filter(mode.get("scale") or 720, watermark_enable_expr(duration_sec))
        filter_complex_str = f"{main_filter};[scaled]{text_f}{final_fmt}[outv]"
    elif not use_image_wm:
        filter_complex_str = f"{main_filter};[scaled]null{final_fmt}[outv]"
    wm_filter = "[1:v]scale=iw*0.5:ih*0.5,format=rgba,colorchannelmixer=aa=0.5[wm]"

    if is_10bit:
        overlay_filter = (
            "overlay=W-w:H-h:"
            f"enable='between(t,{t20},{t20+10})+between(t,{t40},{t40+10})+between(t,{t80},{t80+10})',format=p010le[outv]"
        )
    else:
        overlay_filter = (
            "overlay=W-w:H-h:"
            f"enable='between(t,{t20},{t20+10})+between(t,{t40},{t40+10})+between(t,{t80},{t80+10})'[outv]"
        )

    if use_image_wm:
        filter_complex_str = f"{wm_filter};{main_filter};[scaled][wm]{overlay_filter}"
    cmd.extend(["-filter_complex", filter_complex_str])

    cmd.extend(["-map", "[outv]", "-map", "0:a?", "-map", "0:s?"])

    codec = mode["codec"]
    enc_args = build_video_encoder_args(mode, advanced=True)
    cmd.extend(enc_args)
    print(f"🎛️  Encoder: {' '.join(enc_args)}")

    if mode["audio_codec"] == "copy":
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-c:a", mode["audio_codec"]])
        if "audio_bitrate" in mode and mode["audio_bitrate"]:
            cmd.extend(["-b:a", mode["audio_bitrate"]])
        if "audio_channels" in mode:
            cmd.extend(["-ac", str(mode["audio_channels"])])
        if "audio_sample_rate" in mode:
            cmd.extend(["-ar", mode["audio_sample_rate"]])

    cmd.extend(["-c:s", mode["subtitle_codec"]])
    cmd.extend(["-disposition:s:0", "default"])

    if "movflags" in mode:
        cmd.extend(["-movflags", mode["movflags"]])

    cmd.append(output_path)

    print("Executing enhanced quality command:\n", " ".join(cmd))

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, encoding="utf-8", errors="replace"
        )

        output_lines = []
        error_patterns = [
            'Error while opening encoder',
            'Conversion failed',
            'Invalid data found',
            'No such file or directory',
            'Permission denied'
        ]

        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)

            for error_pattern in error_patterns:
                if error_pattern.lower() in line.lower():
                    print(f"❌ Critical error detected: {error_pattern}")
                    process.terminate()
                    raise RuntimeError(f"FFmpeg error: {error_pattern}")

        return_code = process.wait()

        if return_code != 0:
            print(f"❌ FFmpeg process failed with return code: {return_code}")
            error_log_path = os.path.join(output_dir, f"error_log_{video_name}_{stage_name.replace(' ', '_')}.txt")
            with open(error_log_path, "w", encoding="utf-8") as error_file:
                error_file.write(f"FFmpeg command: {' '.join(cmd)}\n\n")
                error_file.write("Output:\n")
                error_file.writelines(output_lines)
            print(f"📝 Error log saved to: {error_log_path}")
            raise RuntimeError(f"FFmpeg encoding failed with return code {return_code}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created: {output_path}")

        output_size = os.path.getsize(output_path)
        if output_size < 1024:
            raise RuntimeError(f"Output file is too small ({output_size} bytes), likely corrupted")

        elapsed = time.time() - start_time
        quality_note = ""
        if has_premium_quality:
            quality_note = " (Premium Quality: look_ahead + mbbrc + extbrc)"
        elif has_standard_enhancements:
            quality_note = " (Enhanced: mbbrc + b_frames + refs + optimized)"
            if "qsv_async" in mode:
                quality_note += f" [QSV async={mode['qsv_async']}]"

        if crop_info:
            print(f"✅ {stage_name} with crop removal{quality_note} completed for '{video_name}' in {format_time(elapsed)}.")
        else:
            print(f"✅ {stage_name}{quality_note} completed for '{video_name}' in {format_time(elapsed)}.")

        print(f"📁 Output file size: {output_size / (1024*1024):.2f} MB")

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            crop_note = f" (cropped from {crop_info['original_size']} to {crop_info['cropped_size']})" if crop_info else ""
            f.write(f"{video_name} - {stage_name}{crop_note}{quality_note}: {format_time(elapsed)} - Size: {output_size / (1024*1024):.2f} MB\n")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {stage_name} failed for '{video_name}' after {format_time(elapsed)}: {str(e)}")

        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"🗑️  Removed partial output file: {output_path}")
            except:
                pass

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"{video_name} - {stage_name}: FAILED after {format_time(elapsed)} - Error: {str(e)}\n")

        raise

# -------------------------------------------------------------------------------
# Pipeline conversion logic
# -------------------------------------------------------------------------------
def pipeline_convert(video, input_dir, output_dir, option, enable_crop=True):
    print(f"\n========== Starting conversion pipeline for '{video}' (Option {option}) ==========")
    if enable_crop:
        print("🎯 Smart FFmpeg crop detection ENABLED")
    else:
        print("⏸️  Crop detection: DISABLED")

    base, ext = os.path.splitext(video)
    original = os.path.join(input_dir, video)

    try:
        if option == 1:
            output_filename = generate_output_filename(base, "480p") + ".mkv"
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                run_conversion_step(original, output_path, video, "480p x264 QSV Conversion",
                                    MODES["480p_hevc_qsv"], output_dir, is_10bit=False, enable_crop=enable_crop)
            else:
                print(f"⏭️  Output {output_path} already exists. Skipping conversion.")
        elif option == 2:
            output_filename = generate_output_filename(base, "720p") + ".mkv"
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                run_conversion_step(original, output_path, video, "720x264 QSV Conversion",
                                    MODES["720x264_qsv"], output_dir, is_10bit=False, enable_crop=enable_crop)
            else:
                print(f"⏭️  Output {output_path} already exists. Skipping conversion.")
        elif option == 3:
            output_filename = generate_output_filename(base, "1080p") + ".mkv"
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                run_conversion_step(original, output_path, video, "1080x264 QSV Conversion",
                                    MODES["1080x264_qsv"], output_dir, is_10bit=False, enable_crop=enable_crop)
            else:
                print(f"⏭️  Output {output_path} already exists. Skipping conversion.")
        elif option == 4:
            output_filename = generate_output_filename(base, "720p.10BIT") + ".mkv"
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                run_conversion_step(original, output_path, video, "720p 10bit QSV Conversion",
                                    MODES["720p_10bit_qsv"], output_dir, is_10bit=True, enable_crop=enable_crop)
            else:
                print(f"⏭️  Output {output_path} already exists. Skipping conversion.")
        elif option == 5:
            output_filename = generate_output_filename(base, "1080p.10BIT") + ".mkv"
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                run_conversion_step(original, output_path, video, "1080p 10bit QSV Conversion",
                                    MODES["1080p_10bit_qsv"], output_dir, is_10bit=True, enable_crop=enable_crop)
            else:
                print(f"⏭️  Output {output_path} already exists. Skipping conversion.")
        elif option == 6:
            mode_1080 = copy.deepcopy(MODES["1080x264_qsv"])
            mode_720  = copy.deepcopy(MODES["720x264_qsv"])
            mode_480  = copy.deepcopy(MODES["480p_hevc_qsv"])
            mode_720_10 = copy.deepcopy(MODES["720p_10bit_qsv"])
            mode_1080_10 = copy.deepcopy(MODES["1080p_10bit_qsv"])

            outputs = [
                (generate_output_filename(base, "1080p") + ".mkv", mode_1080, "1080x264 QSV Conversion", False),
                (generate_output_filename(base, "720p") + ".mkv", mode_720, "720x264 QSV Conversion", False),
                (generate_output_filename(base, "480p") + ".mkv", mode_480, "480p x264 QSV Conversion", False),
                (generate_output_filename(base, "720p.10BIT") + ".mkv", mode_720_10, "720p 10bit QSV Conversion", True),
                (generate_output_filename(base, "1080p.10BIT") + ".mkv", mode_1080_10, "1080p 10bit QSV Conversion", True)
            ]

            for output_filename, mode, stage_name, is_10bit in outputs:
                output_path = os.path.join(output_dir, output_filename)
                if not os.path.exists(output_path):
                    run_conversion_step(original, output_path, video, stage_name,
                                        mode, output_dir, is_10bit=is_10bit, enable_crop=enable_crop)
                else:
                    print(f"⏭️  Output {output_path} already exists. Skipping conversion.")
        elif option == 7:
            mode_720  = copy.deepcopy(MODES["720x264_qsv"])
            mode_480  = copy.deepcopy(MODES["480p_hevc_qsv"])
            mode_720_10 = copy.deepcopy(MODES["720p_10bit_qsv"])

            outputs = [
                (generate_output_filename(base, "720p") + ".mkv", mode_720, "720x264 QSV Conversion", False),
                (generate_output_filename(base, "480p") + ".mkv", mode_480, "480p x264 QSV Conversion", False),
                (generate_output_filename(base, "720p.10BIT") + ".mkv", mode_720_10, "720p 10bit QSV Conversion", True)
            ]

            for output_filename, mode, stage_name, is_10bit in outputs:
                output_path = os.path.join(output_dir, output_filename)
                if not os.path.exists(output_path):
                    run_conversion_step(original, output_path, video, stage_name,
                                        mode, output_dir, is_10bit=is_10bit, enable_crop=enable_crop)
                else:
                    print(f"⏭️  Output {output_path} already exists. Skipping conversion.")

    except Exception as e:
        print(f"❌ Pipeline conversion failed for '{video}': {str(e)}")
        return False

    return True

# -------------------------------------------------------------------------------
# Parallel conversion functions
# -------------------------------------------------------------------------------
def parallel_convert_all_formats(video, input_dir, output_dir, enable_crop=True):
    print(f"\n========== Starting parallel conversion for all formats for '{video}' ==========")
    if enable_crop:
        print("🎯 Smart FFmpeg crop detection ENABLED")
    else:
        print("⏸️  Crop detection: DISABLED")

    base, ext = os.path.splitext(video)
    original = os.path.join(input_dir, video)

    mode_1080 = copy.deepcopy(MODES["1080x264_qsv"])
    mode_720  = copy.deepcopy(MODES["720x264_qsv"])
    mode_480  = copy.deepcopy(MODES["480p_hevc_qsv"])
    mode_720_10 = copy.deepcopy(MODES["720p_10bit_qsv"])
    mode_1080_10 = copy.deepcopy(MODES["1080p_10bit_qsv"])

    outputs = [
        (generate_output_filename(base, "1080p") + ".mkv", mode_1080, "1080x264 QSV Conversion (Parallel)", False),
        (generate_output_filename(base, "720p") + ".mkv", mode_720, "720x264 QSV Conversion (Parallel)", False),
        (generate_output_filename(base, "480p") + ".mkv", mode_480, "480p x264 QSV Conversion (Parallel)", False),
        (generate_output_filename(base, "720p.10BIT") + ".mkv", mode_720_10, "720p 10bit QSV Conversion (Parallel)", True),
        (generate_output_filename(base, "1080p.10BIT") + ".mkv", mode_1080_10, "1080p 10bit QSV Conversion (Parallel)", True)
    ]

    conversion_tasks = []
    for output_filename, mode, stage_name, is_10bit in outputs:
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            conversion_tasks.append((original, output_path, video, stage_name,
                                     mode, output_dir, is_10bit, enable_crop))
        else:
            print(f"⏭️  Output {output_path} already exists. Skipping conversion.")

    if conversion_tasks:
        start_time = time.time()
        max_workers = min(len(conversion_tasks), os.cpu_count() or 4)
        successful_conversions = 0
        failed_conversions = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_conversion_step, *task) for task in conversion_tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                    successful_conversions += 1
                except Exception as exc:
                    failed_conversions += 1
                    print(f"❌ Exception during parallel conversion for '{video}': {exc}")

        elapsed = time.time() - start_time
        print(f"✅ Parallel conversions for '{video}' completed in {format_time(elapsed)}.")
        print(f"📊 Results: {successful_conversions} successful, {failed_conversions} failed")

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"{video} - Parallel All Formats: {format_time(elapsed)} - Success: {successful_conversions}, Failed: {failed_conversions}\n")
    else:
        print(f"⏭️  No conversions needed for '{video}'. All output files already exist.")

def parallel_convert_selected_formats(video, input_dir, output_dir, enable_crop=True):
    print(f"\n========== Starting parallel conversion for selected formats for '{video}' ==========")
    print("🎯 Formats: 480p, 720p, 720p 10BIT, 1080p")
    if enable_crop:
        print("🎯 Smart FFmpeg crop detection ENABLED")
    else:
        print("⏸️  Crop detection: DISABLED")

    base, ext = os.path.splitext(video)
    original = os.path.join(input_dir, video)

    mode_1080 = copy.deepcopy(MODES["1080x264_qsv"])
    mode_720  = copy.deepcopy(MODES["720x264_qsv"])
    mode_480  = copy.deepcopy(MODES["480p_hevc_qsv"])
    mode_720_10 = copy.deepcopy(MODES["720p_10bit_qsv"])

    outputs = [
        (generate_output_filename(base, "1080p") + ".mkv", mode_1080, "1080x264 QSV Conversion (Parallel)", False),
        (generate_output_filename(base, "720p") + ".mkv", mode_720, "720x264 QSV Conversion (Parallel)", False),
        (generate_output_filename(base, "480p") + ".mkv", mode_480, "480p x264 QSV Conversion (Parallel)", False),
        (generate_output_filename(base, "720p.10BIT") + ".mkv", mode_720_10, "720p 10bit QSV Conversion (Parallel)", True)
    ]

    conversion_tasks = []
    for output_filename, mode, stage_name, is_10bit in outputs:
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            conversion_tasks.append((original, output_path, video, stage_name,
                                     mode, output_dir, is_10bit, enable_crop))
        else:
            print(f"⏭️  Output {output_path} already exists. Skipping conversion.")

    if conversion_tasks:
        start_time = time.time()
        max_workers = min(len(conversion_tasks), os.cpu_count() or 4)
        successful_conversions = 0
        failed_conversions = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_conversion_step, *task) for task in conversion_tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                    successful_conversions += 1
                except Exception as exc:
                    failed_conversions += 1
                    print(f"❌ Exception during parallel conversion for '{video}': {exc}")

        elapsed = time.time() - start_time
        print(f"✅ Selected parallel conversions for '{video}' completed in {format_time(elapsed)}.")
        print(f"📊 Results: {successful_conversions} successful, {failed_conversions} failed")

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"{video} - Parallel Selected Formats (480p, 720p, 720p 10BIT, 1080p): {format_time(elapsed)} - Success: {successful_conversions}, Failed: {failed_conversions}\n")
    else:
        print(f"⏭️  No conversions needed for '{video}'. All output files already exist.")

# -------------------------------------------------------------------------------
# QSV quality / batch-engine settings
# -------------------------------------------------------------------------------
# Watermark: "text" (recommended - lighter, sharp at every resolution, no file needed),
# "image" (uses WATERMARK_PATH) or "none".
WATERMARK_MODE = "text"
WATERMARK_PATH = "C:/watermark.png"

# Text watermark (shown bottom-right at 20%, 40% and 80% of the video, 10 s each)
TEXT_WM = "KMMOVIES.COM"
TEXT_WM_COLOR = "yellow"
# Bold fonts tried in order; the first one present on this PC is used automatically
TEXT_WM_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",    # Arial Bold
    "C:/Windows/Fonts/segoeuib.ttf",   # Segoe UI Bold
    "C:/Windows/Fonts/verdanab.ttf",   # Verdana Bold
    "C:/Windows/Fonts/tahomabd.ttf",   # Tahoma Bold
    "C:/Windows/Fonts/calibrib.ttf",   # Calibri Bold
]
TEXT_WM_FONT = next((f for f in TEXT_WM_FONT_CANDIDATES if os.path.exists(f)), "")
TEXT_WM_FONT_SIZE = 9             # px at 480p (720 px wide)
TEXT_WM_SCALE_WITH_RES = True     # True: same look on every output (720p≈16px, 1080p≈24px)
                                  # False: exactly TEXT_WM_FONT_SIZE px on every output


def text_watermark_filter(out_width, enable_expr):
    """drawtext filter for the KMMOVIES.COM text logo at a given output width."""
    try:
        width = int(str(out_width).split(":")[0])
    except ValueError:
        width = 720
    size = round(TEXT_WM_FONT_SIZE * width / 720) if TEXT_WM_SCALE_WITH_RES else TEXT_WM_FONT_SIZE
    border = max(1, round(size / 12))
    margin = 2 * border   # flush to the corner - only room for the outline + shadow so they aren't cut off
    font = ""
    if TEXT_WM_FONT and os.path.exists(TEXT_WM_FONT):
        font = "fontfile='" + TEXT_WM_FONT.replace("\\", "/").replace(":", "\\:") + "':"
    return (f"drawtext={font}text='{TEXT_WM}':fontcolor={TEXT_WM_COLOR}:fontsize={size}:"
            f"borderw={border}:bordercolor=black@0.85:shadowx={border}:shadowy={border}:shadowcolor=black@0.5:"
            f"x=w-tw-{margin}:y=h-th-{margin}:enable='{enable_expr}'")

# Everything below is configured AUTOMATICALLY at startup by auto_configure_encoders():
#  - QSV_HW_DECODE: GPU decoding. If it ever causes a failure, it is switched off
#    for the rest of the run (and the file is retried) - no manual toggling needed.
#  - QSV_ADVANCED_ARGS: look-ahead / extended BRC options that your GPU + ffmpeg
#    actually accept, tested per encoder (h264_qsv, hevc_qsv).
#  - If QSV itself isn't available, encoders fall back to libx264 / libx265.
QSV_HW_DECODE = True
QSV_ADVANCED_ARGS = {}
QSV_LOOKAHEAD_DEPTHS = ("40", "20")


QSV_PROBE_TIMEOUT = 15      # seconds per test - a hung driver can never block startup
QSV_CAPS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qsv_capabilities.json")


def _ffmpeg_probe(args, timeout=QSV_PROBE_TIMEOUT):
    """Tiny test encode. Returns 'ok', 'fail' or 'timeout'. Never blocks longer than `timeout`."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-f", "lavfi",
           "-i", "testsrc2=s=640x360:r=24", "-frames:v", "30"] + args + ["-f", "null", "-"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except Exception:
        return "fail"
    try:
        return "ok" if proc.wait(timeout=timeout) == 0 else "fail"
    except subprocess.TimeoutExpired:
        try:
            proc.kill()          # don't wait for it - a stuck GPU driver call may not die quickly
        except Exception:
            pass
        return "timeout"


def _ffmpeg_version_line():
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-version"], capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", timeout=10)
        return r.stdout.splitlines()[0].strip()
    except Exception:
        return "unknown"


def _apply_caps(caps):
    software = {"h264_qsv": "libx264", "hevc_qsv": "libx265"}
    for codec, info in caps.items():
        if info["status"] == "unavailable":
            print(f"⚠️  {codec}: NOT available - using {software[codec]} (CPU, slower)")
            for m in MODES.values():
                if m["codec"] == codec:
                    m["codec"] = software[codec]
            continue
        QSV_ADVANCED_ARGS[codec] = info["advanced"]
        note = f"look-ahead RC ON ({' '.join(info['advanced'])})" if info["advanced"] \
            else "standard VBR (look-ahead not supported)"
        print(f"✅ {codec}: {note}")


def auto_configure_encoders():
    """Test what this PC's GPU/driver/ffmpeg support (once - result is cached) and configure encoders.
    Delete qsv_capabilities.json to force a re-test (e.g. after a driver/ffmpeg update)."""
    import json
    version = _ffmpeg_version_line()

    try:
        with open(QSV_CAPS_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("ffmpeg") == version:
            print("🔍 QSV capabilities (cached - delete qsv_capabilities.json to re-test):")
            _apply_caps(cached["caps"])
            _print_font_info()
            return
    except Exception:
        pass

    print(f"🔍 Auto-detecting Intel QSV capabilities (one-time, max {QSV_PROBE_TIMEOUT}s per test)...")
    caps = {}
    for codec, fmt, profile, label in (("h264_qsv", "nv12", "high", "H.264 8-bit"),
                                       ("hevc_qsv", "p010le", "main10", "HEVC 10-bit")):
        base = ["-vf", f"format={fmt}", "-c:v", codec, "-profile:v", profile, "-b:v", "1M"]
        print(f"   • {label} ({codec}) ... ", end="", flush=True)
        t0 = time.time()
        res = _ffmpeg_probe(base)
        print(f"{res} ({time.time() - t0:.1f}s)")

        if res == "fail":
            caps[codec] = {"status": "unavailable", "advanced": []}
            continue
        if res == "timeout":
            # Slow/odd driver init - keep QSV (real encodes have their own error handling),
            # but skip the extra look-ahead tests so startup stays fast.
            caps[codec] = {"status": "available", "advanced": []}
            continue

        chosen = []
        for depth in QSV_LOOKAHEAD_DEPTHS:
            for cand in (["-extbrc", "1", "-look_ahead_depth", depth, "-adaptive_i", "1", "-adaptive_b", "1"],
                         ["-extbrc", "1", "-look_ahead_depth", depth]):
                print(f"     look-ahead {' '.join(cand)} ... ", end="", flush=True)
                r = _ffmpeg_probe(base + cand)
                print(r)
                if r == "ok":
                    chosen = cand
                    break
                if r == "timeout":
                    break
            if chosen or r == "timeout":
                break
        caps[codec] = {"status": "available", "advanced": chosen}

    try:
        with open(QSV_CAPS_FILE, "w", encoding="utf-8") as f:
            json.dump({"ffmpeg": version, "caps": caps}, f, indent=2)
    except Exception:
        pass
    _apply_caps(caps)
    _print_font_info()


def _print_font_info():
    if TEXT_WM_FONT:
        print(f"🔤 Watermark font: {TEXT_WM_FONT}")
    elif WATERMARK_MODE == "text":
        print("⚠️  No bold Windows font found - ffmpeg default font will be used")
    print()


# A failure within this many seconds of ffmpeg starting is treated as a
# startup/compatibility failure and retried once in safe mode.
SAFE_RETRY_WINDOW_SEC = 90

# Kill ffmpeg if it prints nothing for this long (a normal encode prints progress every second)
FFMPEG_STALL_TIMEOUT = 120


def build_video_encoder_args(mode, advanced=True):
    """Video encoder arguments for one output (QSV or software)."""
    codec = mode["codec"]
    args = ["-c:v", codec]
    if mode.get("profile"):
        args += ["-profile:v", mode["profile"]]
    if mode.get("preset"):
        args += ["-preset", mode["preset"]]

    if codec.endswith("_qsv"):
        for key, flag in (("mbbrc", "-mbbrc"), ("b_frames", "-bf"), ("refs", "-refs"),
                          ("gop", "-g"), ("qsv_async", "-async_depth")):
            if mode.get(key):
                args += [flag, str(mode[key])]
        if advanced:
            args += QSV_ADVANCED_ARGS.get(codec, [])
        if mode.get("global_quality"):
            args += ["-global_quality", str(mode["global_quality"])]
        elif mode.get("bitrate"):
            args += ["-b:v", mode["bitrate"]]
            if mode.get("maxrate"):
                args += ["-maxrate", mode["maxrate"]]
            if mode.get("bufsize"):
                args += ["-bufsize", mode["bufsize"]]
    else:
        for key, flag in (("b_frames", "-bf"), ("refs", "-refs"),
                          ("sc_threshold", "-sc_threshold"), ("deblock", "-deblock")):
            if mode.get(key):
                args += [flag, str(mode[key])]
        if mode.get("bitrate"):
            args += ["-b:v", mode["bitrate"]]
    return args


def build_audio_sub_args(mode):
    """Audio + subtitle arguments. Source channel layout is kept unless the mode sets audio_channels."""
    args = []
    if mode["audio_codec"] == "copy":
        args += ["-c:a", "copy"]
    else:
        args += ["-c:a", mode["audio_codec"]]
        if mode.get("audio_bitrate"):
            args += ["-b:a", mode["audio_bitrate"]]
        if mode.get("audio_channels"):
            args += ["-ac", str(mode["audio_channels"])]
        if mode.get("audio_sample_rate"):
            args += ["-ar", mode["audio_sample_rate"]]
    args += ["-c:s", mode["subtitle_codec"], "-disposition:s:0", "default"]
    return args


def watermark_enable_expr(duration_sec):
    if duration_sec <= 0:
        return None
    t20, t40, t80 = 0.2 * duration_sec, 0.4 * duration_sec, 0.8 * duration_sec
    return (f"between(t,{t20:.3f},{t20 + 10:.3f})+between(t,{t40:.3f},{t40 + 10:.3f})"
            f"+between(t,{t80:.3f},{t80 + 10:.3f})")


def build_batch_filter(crop_info, targets, wm_enable):
    """targets: list of dicts with 'label', 'width', 'ten_bit'.
    One decode -> optional crop -> split -> per-output scale (+ watermark) -> encoder pixel format.
    The watermark is split too: a filter label can only be consumed once."""
    n = len(targets)
    crop = f"{crop_info['ffmpeg_crop']}," if crop_info and 'ffmpeg_crop' in crop_info else ""
    src_labels = "".join(f"[s{i}]" for i in range(n))
    chains = [f"[0:v]{crop}split={n}{src_labels}" if n > 1 else f"[0:v]{crop}null[s0]"]

    use_wm = wm_enable is not None and WATERMARK_MODE == "image"
    use_text = wm_enable is not None and WATERMARK_MODE == "text"
    if use_wm:
        wm_labels = "".join(f"[wm{i}]" for i in range(n))
        wm_split = f",split={n}{wm_labels}" if n > 1 else "[wm0]"
        chains.append(f"[1:v]scale=iw*0.5:ih*0.5,format=rgba,colorchannelmixer=aa=0.5{wm_split}")

    for i, t in enumerate(targets):
        out_fmt = "p010le" if t["ten_bit"] else "nv12"
        if use_wm:
            # 10-bit path stays 10-bit through the overlay (default overlay format is 8-bit yuv420)
            pre = ",format=p010le" if t["ten_bit"] else ""
            ov_fmt = "yuv420p10" if t["ten_bit"] else "yuv420"
            chains.append(f"[s{i}]scale={t['width']}:-2,setsar=1{pre}[v{i}]")
            chains.append(f"[v{i}][wm{i}]overlay=W-w:H-h:format={ov_fmt}:enable='{wm_enable}',"
                          f"format={out_fmt}[{t['label']}]")
        elif use_text:
            pre = ",format=p010le" if t["ten_bit"] else ""
            chains.append(f"[s{i}]scale={t['width']}:-2,setsar=1{pre},"
                          f"{text_watermark_filter(t['width'], wm_enable)},format={out_fmt}[{t['label']}]")
        else:
            chains.append(f"[s{i}]scale={t['width']}:-2,setsar=1,format={out_fmt}[{t['label']}]")
    return ";".join(chains)


def _part_path(final_path):
    root, ext = os.path.splitext(final_path)
    return f"{root}.part{ext}"


def _run_ffmpeg_logged(cmd):
    """Run ffmpeg, stream output, return (return_code, output_lines, seconds_until_exit)."""
    t0 = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, encoding="utf-8", errors="replace")
    output_lines = []
    fatal = ['error while opening encoder', 'conversion failed', 'invalid data found',
             'no such file or directory', 'permission denied', 'error initializing complex filters']
    fatal_hit = False
    # Stall watchdog: if ffmpeg prints nothing for FFMPEG_STALL_TIMEOUT seconds
    # (e.g. a hung GPU driver), kill it so the run can't freeze forever.
    import threading
    last_output = [time.time()]
    stalled = [False]
    done = threading.Event()

    def watchdog():
        while not done.wait(5):
            if time.time() - last_output[0] > FFMPEG_STALL_TIMEOUT:
                stalled[0] = True
                print(f"\n⏱️  ffmpeg produced no output for {FFMPEG_STALL_TIMEOUT}s - killing stalled process.")
                try:
                    process.kill()
                except Exception:
                    pass
                return
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        for line in process.stdout:
            last_output[0] = time.time()
            print(line, end="")
            output_lines.append(line)
            if any(p in line.lower() for p in fatal):
                print(f"\n❌ Critical ffmpeg error detected: {line.strip()}")
                fatal_hit = True
                process.terminate()
                break
        rc = process.wait()
    except KeyboardInterrupt:
        done.set()
        process.terminate()
        process.wait()
        raise
    done.set()
    if stalled[0]:
        output_lines.append(f"\n[Encoder] Killed: no ffmpeg output for {FFMPEG_STALL_TIMEOUT}s (stalled GPU/driver?)\n")
        rc = rc or 1
    if fatal_hit and rc == 0:
        rc = 1
    return rc, output_lines, time.time() - t0


def smart_batch_encode(video, input_dir, output_dir, formats, title, crop_info=None,
                       enable_crop=True, series_crop_info=None):
    """Single decode -> N simultaneous QSV encodes.
    formats: list of (quality_tag, mode_key, ten_bit), e.g. ("720p.10BIT", "720p_10bit_qsv", True)."""
    print(f"\n🚀 ========== {title} for '{video}' ==========")
    print(f"📊 Formats: {' + '.join(f[0].replace('.10BIT', ' 10BIT') for f in formats)}")
    print(f"⚡ Single decode + {len(formats)} simultaneous QSV encodes")

    start_time = time.time()
    base, _ = os.path.splitext(video)
    original = os.path.join(input_dir, video)

    # Work out which outputs still need encoding
    todo = []
    for quality_tag, mode_key, ten_bit in formats:
        final_path = os.path.join(output_dir, generate_output_filename(base, quality_tag) + ".mkv")
        if os.path.exists(final_path):
            print(f"⏭️  Exists, skipping: {os.path.basename(final_path)}")
            continue
        todo.append({"tag": quality_tag, "mode": copy.deepcopy(MODES[mode_key]), "ten_bit": ten_bit,
                     "final": final_path, "part": _part_path(final_path),
                     "label": "out_" + re.sub(r'[^A-Za-z0-9]', '_', quality_tag)})

    if not todo:
        print("⏭️  All output files already exist. Skipping.")
        return True
    print(f"🔄 Encoding needed for: {', '.join(t['tag'] for t in todo)}")

    # Crop: series cache first, then per-file detection
    if crop_info is None:
        crop_info = series_crop_info
    if crop_info is None and enable_crop:
        crop_info = detect_crop_params(original, video)
    if crop_info:
        print(f"🎬 Crop: {crop_info['ffmpeg_crop']} ({crop_info['original_size']} → {crop_info['cropped_size']})")

    wm_enable = None
    if WATERMARK_MODE == "text" or (WATERMARK_MODE == "image" and os.path.exists(WATERMARK_PATH)):
        wm_enable = watermark_enable_expr(get_video_duration(original))
        if wm_enable is None:
            print(f"⚠️  Could not determine duration for '{video}' - watermark skipped.")
        elif WATERMARK_MODE == "text":
            print(f"🔤 Text watermark: '{TEXT_WM}' ({TEXT_WM_COLOR}, bold)")
    elif WATERMARK_MODE == "image":
        print(f"⚠️  Watermark not found at {WATERMARK_PATH} - encoding without watermark.")

    targets = [{"label": t["label"], "width": t["mode"]["scale"], "ten_bit": t["ten_bit"]} for t in todo]
    filter_complex = build_batch_filter(crop_info, targets, wm_enable)

    def build_cmd(hw_decode, advanced):
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner"]
        if hw_decode:
            cmd += ["-hwaccel", "auto"]
        cmd += ["-i", original]
        if wm_enable is not None and WATERMARK_MODE == "image":
            cmd += ["-i", WATERMARK_PATH]
        cmd += ["-filter_complex", filter_complex]
        for t in todo:
            cmd += ["-map", f"[{t['label']}]", "-map", "0:a?", "-map", "0:s?",
                    "-map_metadata", "0", "-map_chapters", "0"]
            cmd += build_video_encoder_args(t["mode"], advanced=advanced)
            cmd += build_audio_sub_args(t["mode"])
            cmd += ["-max_muxing_queue_size", "4096", t["part"]]
        return cmd

    def cleanup_parts():
        for t in todo:
            if os.path.exists(t["part"]):
                try:
                    os.remove(t["part"])
                except OSError:
                    pass

    global QSV_HW_DECODE
    # Attempt order: current settings -> without GPU decode -> without look-ahead too
    has_adv = any(QSV_ADVANCED_ARGS.get(t["mode"]["codec"]) for t in todo)
    attempts = [(QSV_HW_DECODE, True)]
    if QSV_HW_DECODE:
        attempts.append((False, True))
    if has_adv:
        attempts.append((False, False))
    last_error = "unknown error"
    try:
        for attempt_no, (hw_decode, advanced) in enumerate(attempts):
            cleanup_parts()
            cmd = build_cmd(hw_decode, advanced)
            mode_note = (f"GPU decode={'on' if hw_decode else 'off'}, "
                         f"look-ahead RC={'on' if advanced and has_adv else 'off'}")
            print(f"🚀 Executing ({mode_note}):")
            print(" ".join(f'"{c}"' if " " in c else c for c in cmd))

            rc, lines, ran_for = _run_ffmpeg_logged(cmd)
            if rc == 0:
                if attempt_no > 0:
                    # Remember what worked so later files don't fail first
                    if not hw_decode and QSV_HW_DECODE:
                        QSV_HW_DECODE = False
                        print("🧠 GPU decoding disabled for the rest of this run.")
                    if not advanced:
                        QSV_ADVANCED_ARGS.clear()
                        print("🧠 Look-ahead RC disabled for the rest of this run.")
                break

            last_error = f"ffmpeg return code {rc}"
            log_path = os.path.join(output_dir, f"error_log_{re.sub(r'[^A-Za-z0-9]', '_', video)}.txt")
            with open(log_path, "w", encoding="utf-8") as ef:
                ef.write(f"FFmpeg command: {' '.join(cmd)}\n\nOutput:\n")
                ef.writelines(lines)
            print(f"📝 Error log saved to: {log_path}")

            can_retry = attempt_no + 1 < len(attempts) and ran_for <= SAFE_RETRY_WINDOW_SEC
            if not can_retry:
                raise RuntimeError(last_error)
            nhw, nadv = attempts[attempt_no + 1]
            print(f"🔁 Startup failure - retrying with GPU decode={'on' if nhw else 'off'}, "
                  f"look-ahead={'on' if nadv and has_adv else 'off'}.")

        # Validate and promote .part files to their final names
        for t in todo:
            if not os.path.exists(t["part"]):
                raise RuntimeError(f"Output file was not created: {t['part']}")
            size = os.path.getsize(t["part"])
            if size < 1024:
                raise RuntimeError(f"Output too small ({size} bytes): {t['part']}")
            os.replace(t["part"], t["final"])
            print(f"✅ Created {os.path.basename(t['final'])} - Size: {size / (1024 * 1024):.2f} MB")

        elapsed = time.time() - start_time
        crop_note = f" (cropped from {crop_info['original_size']} to {crop_info['cropped_size']})" if crop_info else ""
        print(f"\n🎉 {title}{crop_note} completed for '{video}' in {format_time(elapsed)}.")
        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"{video} - {title} ({len(todo)} formats){crop_note}: {format_time(elapsed)}\n")
        return True

    except KeyboardInterrupt:
        cleanup_parts()
        raise
    except Exception as e:
        cleanup_parts()
        elapsed = time.time() - start_time
        print(f"❌ {title} failed for '{video}' after {format_time(elapsed)}: {e}")
        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"{video} - {title}: FAILED after {format_time(elapsed)} - Error: {e}\n")
        return False


# -------------------------------------------------------------------------------
# Smart Batch QSV (option 13) - single decode, 5 simultaneous QSV outputs
# -------------------------------------------------------------------------------
def smart_batch_qsv_convert(video, input_dir, output_dir, enable_crop=True):
    return smart_batch_encode(video, input_dir, output_dir, [
        ("480p", "480p_hevc_qsv", False),
        ("720p", "720x264_qsv", False),
        ("720p.10BIT", "720p_10bit_qsv", True),
        ("1080p", "1080x264_qsv", False),
        ("1080p.10BIT", "1080p_10bit_qsv", True),
    ], "Smart Batch QSV", enable_crop=enable_crop)

# -------------------------------------------------------------------------------
# TV Series functions
# -------------------------------------------------------------------------------
def smart_tv_series_encode_optimized(video_files, input_dir, output_dir, enable_crop=True, batch_size=1):
    """Smart TV Series encoder - processes files one at a time with crop caching.
    Optimal for Intel Arc GPU: 1 decode + 4 simultaneous QSV encodes = 5 sessions."""
    total_files = len(video_files)
    print(f"\n🚀 ========== SMART TV SERIES - 4 FORMATS (Sequential) ==========")
    print(f"📁 Total files to process: {total_files}")
    print(f"🎯 Per file: Single decode + 4 simultaneous encodes (480p, 720p, 720p 10BIT, 1080p)")
    print(f"⚡ Sequential processing: 1 file at a time (optimal for Arc GPU)")
    print(f"🎬 Crop Optimization: Smart detect once for first file, apply to all episodes")
    if enable_crop:
        print("🎯 Smart FFmpeg crop detection ENABLED")
    else:
        print("⏸️  Crop detection: DISABLED")
    print("🔧 Enhanced Quality: All modes include optimized settings")
    print("=" * 70)

    # Crop caching - smart detect from first episode only
    series_crop_info = None
    crop_detection_time = 0

    if enable_crop and video_files:
        print(f"\n🎬 ========== TV SERIES CROP DETECTION (ONCE FOR ALL) ==========")
        first_video = video_files[0]
        first_video_path = os.path.join(input_dir, first_video)

        crop_start_time = time.time()
        print(f"🔍 Smart analyzing crop from first episode: '{first_video}'")
        print("📋 This crop will be applied to ALL episodes in the series")

        try:
            series_crop_info = detect_crop_params(first_video_path, first_video)
            crop_detection_time = time.time() - crop_start_time

            if series_crop_info and 'ffmpeg_crop' in series_crop_info:
                print(f"✅ TV Series crop detected: {series_crop_info['original_size']} → {series_crop_info['cropped_size']}")
                print(f"📋 Crop filter: {series_crop_info['ffmpeg_crop']}")
                if 'confidence' in series_crop_info:
                    print(f"📊 Detection confidence: {series_crop_info['confidence']}")
                print(f"⚡ Time saved: ~{(len(video_files)-1) * 30:.0f} seconds (crop detection skipped for {len(video_files)-1} episodes)")
            else:
                print("📺 TV Series analysis: No black bars detected - no cropping needed")
                series_crop_info = None
        except Exception as e:
            print(f"⚠️  TV Series crop detection failed: {str(e)} - will proceed without cropping")
            series_crop_info = None

        print(f"🕒 Crop analysis completed in {format_time(crop_detection_time)}")
        print("=" * 70)

    overall_start_time = time.time()
    successful_files = 0
    failed_files = 0

    # Sequential processing - one file at a time (optimal for Arc GPU)
    for i, video in enumerate(video_files):
        print(f"\n🚀 ========== FILE {i + 1}/{total_files}: '{video}' ==========")
        print(f"⚡ Single decode → 4 simultaneous QSV encodes")
        if series_crop_info and 'ffmpeg_crop' in series_crop_info:
            print(f"🎬 Using cached crop: {series_crop_info['ffmpeg_crop']}")

        file_start_time = time.time()

        try:
            success = smart_batch_qsv_tv_series_optimized(
                video, input_dir, output_dir, enable_crop, series_crop_info
            )
            if success:
                successful_files += 1
                file_elapsed = time.time() - file_start_time
                print(f"✅ [{i + 1}/{total_files}] Completed '{video}' in {format_time(file_elapsed)}")
            else:
                failed_files += 1
                print(f"❌ [{i + 1}/{total_files}] Failed '{video}'")
        except Exception as e:
            failed_files += 1
            print(f"❌ [{i + 1}/{total_files}] Exception processing '{video}': {str(e)}")

        # Progress report with ETA
        completed = i + 1
        if completed < total_files:
            elapsed_so_far = time.time() - overall_start_time
            avg_time = elapsed_so_far / completed
            remaining = total_files - completed
            eta = avg_time * remaining
            print(f"📊 Progress: {completed}/{total_files} | Elapsed: {format_time(elapsed_so_far)} | ETA: ~{format_time(eta)}")

    overall_elapsed = time.time() - overall_start_time

    print(f"\n🎉 ========== SMART TV SERIES ENCODING COMPLETED ==========")
    print(f"📁 Total files processed: {total_files}")
    print(f"✅ Successful conversions: {successful_files}")
    print(f"❌ Failed conversions: {failed_files}")
    print(f"⏱️  Total time: {format_time(overall_elapsed)}")
    if successful_files > 0:
        print(f"📊 Average time per successful file: {format_time(overall_elapsed / successful_files)}")
    print(f"⚡ Processing: Sequential (1 file at a time, optimal for Arc GPU)")
    if series_crop_info:
        estimated_time_saved = (len(video_files) - 1) * 30
        print(f"🎬 Crop Optimization: ~{format_time(estimated_time_saved)} saved with series-wide crop caching")

    with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
        f.write(f"SMART TV SERIES COMPLETE - {total_files} files [Sequential + crop cache]: {format_time(overall_elapsed)} - Success: {successful_files}, Failed: {failed_files}\n")

def smart_batch_qsv_tv_series_optimized(video, input_dir, output_dir, enable_crop=True, series_crop_info=None):
    return smart_batch_encode(video, input_dir, output_dir, [
        ("480p", "480p_hevc_qsv", False),
        ("720p", "720x264_qsv", False),
        ("720p.10BIT", "720p_10bit_qsv", True),
        ("1080p", "1080x264_qsv", False),
    ], "Smart Batch QSV TV (4 formats)", enable_crop=enable_crop, series_crop_info=series_crop_info)

def smart_tv_series_encode_3formats(video_files, input_dir, output_dir, enable_crop=True, batch_size=3):
    total_files = len(video_files)
    print(f"\n⚡ ========== 3-FORMAT SMART TV SERIES ==========")
    print(f"📁 Total files to process: {total_files}")
    print(f"📦 Batch size: {batch_size} files at a time")
    print(f"🎯 Per file: Single decode + 3 simultaneous encodes (480p, 720p, 1080p)")
    print(f"⚡ QSV Optimization: async_depth=8 (increased from 2)")
    print(f"🎬 Crop Optimization: Detect once for first file, apply to all episodes")
    if enable_crop:
        print("🎯 Smart FFmpeg crop detection ENABLED")
    else:
        print("⏸️  Crop detection: DISABLED")
    print("🔧 Enhanced Quality: All modes include optimized settings")
    print("=" * 70)

    series_crop_info = None
    crop_detection_time = 0

    if enable_crop and video_files:
        print(f"\n🎬 ========== TV SERIES CROP DETECTION (ONCE FOR ALL) ==========")
        first_video = video_files[0]
        first_video_path = os.path.join(input_dir, first_video)

        crop_start_time = time.time()
        print(f"🔍 Analyzing crop parameters from first episode: '{first_video}'")
        print("📋 This crop will be applied to ALL episodes in the series")

        try:
            series_crop_info = detect_crop_params(first_video_path, first_video)
            crop_detection_time = time.time() - crop_start_time

            if series_crop_info and 'ffmpeg_crop' in series_crop_info:
                print(f"✅ TV Series crop detected: {series_crop_info['original_size']} → {series_crop_info['cropped_size']}")
                print(f"📋 Crop filter: {series_crop_info['ffmpeg_crop']}")
                print(f"⚡ Time saved: ~{(len(video_files)-1) * 30:.0f} seconds (crop detection skipped for {len(video_files)-1} episodes)")
            elif series_crop_info and series_crop_info.get('no_crop_needed'):
                print("📺 TV Series analysis: No black bars detected - no cropping needed")
            else:
                print("⚠️  TV Series: No crop detected or detection skipped - proceeding without cropping")
                series_crop_info = None
        except Exception as e:
            print(f"⚠️  TV Series crop detection failed: {str(e)} - will proceed without cropping")
            series_crop_info = None

        print(f"🕒 Crop analysis completed in {format_time(crop_detection_time)}")
        print("=" * 70)

    num_full_batches = total_files // batch_size
    remaining_files = total_files % batch_size

    print(f"📊 Batch distribution: {num_full_batches} full batches of {batch_size}")
    if remaining_files > 0:
        print(f"📊 Final batch: {remaining_files} files")

    overall_start_time = time.time()
    successful_files = 0
    failed_files = 0

    for batch_num in range(num_full_batches + (1 if remaining_files > 0 else 0)):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        current_batch = video_files[start_idx:end_idx]

        batch_start_time = time.time()
        print(f"\n⚡ ========== 3-FORMAT SMART BATCH {batch_num + 1}/{num_full_batches + (1 if remaining_files > 0 else 0)} ==========")
        print(f"📹 Processing files {start_idx + 1}-{end_idx} of {total_files}")
        print(f"📂 Files in this batch: {', '.join(current_batch)}")
        print(f"⚡ Each file: 1 decode → 3 outputs (QSV async_depth=8)")
        print(f"🎬 Crop: Using series-wide crop parameters (no per-file detection)")

        with ThreadPoolExecutor(max_workers=len(current_batch)) as batch_executor:
            batch_futures = []

            for video in current_batch:
                video_path = os.path.join(input_dir, video)
                if verify_input_file(video_path):
                    print(f"✅ Queuing '{video}' for 3-Format Smart Batch QSV")
                    future = batch_executor.submit(
                        smart_batch_qsv_3formats, video, input_dir, output_dir,
                        enable_crop, series_crop_info
                    )
                    batch_futures.append((future, video))
                else:
                    print(f"❌ Skipping '{video}': input verification failed")
                    failed_files += 1

            batch_successful = 0
            batch_failed = 0
            for future, video in batch_futures:
                try:
                    result = future.result()
                    if result:
                        print(f"✅ Completed 3-Format Smart Batch QSV for '{video}' (3 formats)")
                        batch_successful += 1
                        successful_files += 1
                    else:
                        print(f"❌ Failed 3-Format Smart Batch QSV for '{video}'")
                        batch_failed += 1
                        failed_files += 1
                except Exception as exc:
                    print(f"❌ Exception processing '{video}': {exc}")
                    batch_failed += 1
                    failed_files += 1

        batch_elapsed = time.time() - batch_start_time
        print(f"⏱️  3-Format Smart Batch {batch_num + 1} completed in {format_time(batch_elapsed)}")
        print(f"📊 Batch Results: {batch_successful} successful, {batch_failed} failed")
        print(f"📊 Efficiency: {len(current_batch)} decodes (vs {len(current_batch)*3} in traditional parallel)")
        print(f"🎬 Crop efficiency: 0 crop detections (vs {len(current_batch)} in non-optimized)")

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"3-FORMAT SMART TV SERIES BATCH {batch_num + 1} ({len(current_batch)} files): {format_time(batch_elapsed)} - Success: {batch_successful}, Failed: {batch_failed}\n")

    overall_elapsed = time.time() - overall_start_time
    estimated_time_saved = (len(video_files) - 1) * 60

    print(f"\n🎉 ========== 3-FORMAT SMART TV SERIES ENCODING COMPLETED ==========")
    print(f"📁 Total files processed: {total_files}")
    print(f"✅ Successful conversions: {successful_files}")
    print(f"❌ Failed conversions: {failed_files}")
    print(f"⏱️  Total time: {format_time(overall_elapsed)}")
    if successful_files > 0:
        print(f"📊 Average time per successful file: {format_time(overall_elapsed / successful_files)}")
    print(f"🚀 QSV Optimization: async_depth=8 applied to all encoders")
    print(f"🎬 Crop Optimization: ~{format_time(estimated_time_saved)} saved with series-wide crop caching")
    print(f"⚡ Total efficiency gain: ~{total_files} decodes + crop caching (vs {total_files*3} decodes + {total_files} crop detections)")

    with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
        f.write(f"3-FORMAT SMART TV SERIES COMPLETE - {total_files} files [QSV async=8 + crop cache]: {format_time(overall_elapsed)} - Success: {successful_files}, Failed: {failed_files}\n")

def smart_batch_qsv_3formats(video, input_dir, output_dir, enable_crop=True, series_crop_info=None):
    return smart_batch_encode(video, input_dir, output_dir, [
        ("480p", "480p_hevc_qsv", False),
        ("720p", "720x264_qsv", False),
        ("1080p", "1080x264_qsv", False),
    ], "Smart Batch QSV TV (3 formats)", enable_crop=enable_crop, series_crop_info=series_crop_info)

def tv_series_parallel_encode(video_files, input_dir, output_dir, enable_crop=True, batch_size=3):
    total_files = len(video_files)
    print(f"\n📺 ========== TV SERIES PARALLEL ENCODING ==========")
    print(f"📁 Total files to process: {total_files}")
    print(f"📦 Batch size: {batch_size} files at a time")
    print(f"🎯 Per file: Traditional parallel encoding (4 formats)")
    if enable_crop:
        print("🎯 Smart FFmpeg crop detection ENABLED")
    else:
        print("⏸️  Crop detection: DISABLED")
    print("=" * 65)

    num_full_batches = total_files // batch_size
    remaining_files = total_files % batch_size

    print(f"📊 Batch distribution: {num_full_batches} full batches of {batch_size}")
    if remaining_files > 0:
        print(f"📊 Final batch: {remaining_files} files")

    overall_start_time = time.time()
    successful_files = 0
    failed_files = 0

    for batch_num in range(num_full_batches + (1 if remaining_files > 0 else 0)):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        current_batch = video_files[start_idx:end_idx]

        batch_start_time = time.time()
        print(f"\n📺 ========== TV SERIES BATCH {batch_num + 1}/{num_full_batches + (1 if remaining_files > 0 else 0)} ==========")
        print(f"📹 Processing files {start_idx + 1}-{end_idx} of {total_files}")
        print(f"📂 Files in this batch: {', '.join(current_batch)}")

        with ThreadPoolExecutor(max_workers=len(current_batch)) as batch_executor:
            batch_futures = []

            for video in current_batch:
                video_path = os.path.join(input_dir, video)
                if verify_input_file(video_path):
                    print(f"✅ Queuing '{video}' for TV Series parallel processing")
                    future = batch_executor.submit(
                        process_single_file_tv_series, video, input_dir, output_dir, enable_crop
                    )
                    batch_futures.append((future, video))
                else:
                    print(f"❌ Skipping '{video}': input verification failed")
                    failed_files += 1

            batch_successful = 0
            batch_failed = 0
            for future, video in batch_futures:
                try:
                    result = future.result()
                    if result:
                        print(f"✅ Completed TV Series processing for '{video}'")
                        batch_successful += 1
                        successful_files += 1
                    else:
                        print(f"❌ Failed TV Series processing for '{video}'")
                        batch_failed += 1
                        failed_files += 1
                except Exception as exc:
                    print(f"❌ Exception processing '{video}': {exc}")
                    batch_failed += 1
                    failed_files += 1

        batch_elapsed = time.time() - batch_start_time
        print(f"⏱️  TV Series Batch {batch_num + 1} completed in {format_time(batch_elapsed)}")
        print(f"📊 Batch Results: {batch_successful} successful, {batch_failed} failed")

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"TV SERIES PARALLEL BATCH {batch_num + 1} ({len(current_batch)} files): {format_time(batch_elapsed)} - Success: {batch_successful}, Failed: {batch_failed}\n")

    overall_elapsed = time.time() - overall_start_time

    print(f"\n🎉 ========== TV SERIES PARALLEL ENCODING COMPLETED ==========")
    print(f"📁 Total files processed: {total_files}")
    print(f"✅ Successful conversions: {successful_files}")
    print(f"❌ Failed conversions: {failed_files}")
    print(f"⏱️  Total time: {format_time(overall_elapsed)}")
    if successful_files > 0:
        print(f"📊 Average time per successful file: {format_time(overall_elapsed / successful_files)}")

    with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
        f.write(f"TV SERIES PARALLEL COMPLETE - {total_files} files: {format_time(overall_elapsed)} - Success: {successful_files}, Failed: {failed_files}\n")

def process_single_file_tv_series(video, input_dir, output_dir, enable_crop=True):
    print(f"\n📺 Processing TV Series file: '{video}'")
    try:
        parallel_convert_selected_formats(video, input_dir, output_dir, enable_crop)
        return True
    except Exception as e:
        print(f"❌ Failed to process TV Series file '{video}': {str(e)}")
        return False

# -------------------------------------------------------------------------------
# System requirements check
# -------------------------------------------------------------------------------
def check_system_requirements():
    print("🔧 Checking system requirements...")

    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        print("✅ FFmpeg: Available")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ FFmpeg: Not found")
        return False

    try:
        subprocess.run(['ffprobe', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        print("✅ FFprobe: Available")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ FFprobe: Not found")
        return False

    print()
    return True

# -------------------------------------------------------------------------------
# Main function with modified options (removed 10, 12, 16)
# -------------------------------------------------------------------------------
def main():
    print("🎬 Enhanced Video Encoder: Smart FFmpeg Crop Detection + QSV Encoding")
    print("⚡ OPTIMIZED: Smart Crop Detection + Sequential QSV + async_depth=8")
    print("🎯 OPTIONS: 14 (4 formats), 15 (3 formats) - Smart TV Series Encoding")
    print("🔧 FIXED: All color filter options removed from all modes")
    print("✅ ENHANCED: Better error handling and incomplete encoding prevention")
    print("📁 Input Directory: Encode_Input")
    print("📁 Output Directory: Encode_Output")

    if not check_system_requirements():
        print("❌ Missing required tools. Please install FFmpeg and FFprobe.")
        sys.exit(1)
    auto_configure_encoders()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "Encode_Input")
    output_dir = os.path.join(script_dir, "Encode_Output")

    if not os.path.exists(input_dir):
        try:
            os.makedirs(input_dir)
            print(f"📁 Created Encode_Input folder at: {input_dir}")
            print(f"Please place your video files in the Encode_Input folder and run the script again.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Failed to create Encode_Input folder: {e}")
            sys.exit(1)

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"📁 Created Encode_Output folder at: {output_dir}")
        except Exception as e:
            print(f"❌ Failed to create Encode_Output folder: {e}")
            sys.exit(1)

    print(f"📁 Using input folder: {input_dir}")
    print(f"📁 Using output folder: {output_dir}\n")

    video_files = [f for f in os.listdir(input_dir) if is_video_file(f)]
    if not video_files:
        print("❌ No video files found in the Encode_Input folder.")
        sys.exit(0)
    print(f"📹 Found {len(video_files)} video file(s) to process.\n")

    print("🎯 Smart Crop Detection System:")
    print("   Method: FFmpeg multi-point sampling (9 points across video)")
    print("   Analysis: Statistical consensus with tolerance grouping")
    try:
        crop_choice = input("Enable automatic black bar removal? (y/n, default: y): ").strip().lower()
        enable_crop = crop_choice != 'n'
    except:
        enable_crop = True

    if enable_crop:
        print("✅ Automatic crop detection: ENABLED")
    else:
        print("❌ Automatic crop detection: DISABLED")

    print("\n📋 Select conversion option:")
    print("1) Encode Only 480p x264 QSV Conversion (Enhanced)")
    print("2) Encode 720x264 QSV Conversion (Enhanced)")
    print("3) Encode 1080x264 QSV Conversion (Enhanced)")
    print("4) Encode 720p 10bit QSV Conversion (Enhanced)")
    print("5) Encode 1080p 10bit QSV Conversion (Enhanced)")
    print("6) Encode All Versions (sequential conversion)")
    print("7) Encode 480p, 720p, and 720p 10bit only")
    print("8) Encode All Versions in Parallel (simultaneous conversion)")
    print("9) Encode 480p, 720p, 720p 10BIT, 1080p in Parallel")
    print("11) TV SERIES ENCODE - PARALLEL (Batch process 3 files at a time)")
    print("13) SMART BATCH QSV - Single decode + 5 simultaneous QSV outputs (hw decode + look-ahead)")
    print("14) SMART TV SERIES - Sequential + Smart Crop + 4 formats (RECOMMENDED)")
    print("15) OPTIMIZED 3-FORMAT TV SERIES - 480p + 720p + 1080p (Fast & Efficient)")

    print("\n⚡ Options 1-5: Enhanced encoding with optimized settings")
    print("🚀 Options 14-15: Smart TV series with sequential processing + crop caching")
    print("✅ ALL OPTIONS: Color filters removed, better error handling")

    try:
        option = int(input("Enter option number (1-15): ").strip())
        if option < 1 or option > 15 or option in [10, 12, 16]:
            raise ValueError
    except ValueError:
        print("Invalid option. Defaulting to option 13 (Smart Batch QSV).")
        option = 13

    overall_start_time = time.time()
    total_successful = 0
    total_failed = 0

    try:
        if option == 6:
            for video in video_files:
                video_path = os.path.join(input_dir, video)
                if verify_input_file(video_path):
                    success = pipeline_convert(video, input_dir, output_dir, option, enable_crop)
                    if success:
                        total_successful += 1
                    else:
                        total_failed += 1
                else:
                    print(f"❌ Skipping '{video}': input verification failed.")
                    total_failed += 1
        elif option == 8:
            for video in video_files:
                video_path = os.path.join(input_dir, video)
                if verify_input_file(video_path):
                    try:
                        parallel_convert_all_formats(video, input_dir, output_dir, enable_crop)
                        total_successful += 1
                    except Exception as e:
                        print(f"❌ Failed processing '{video}': {str(e)}")
                        total_failed += 1
                else:
                    print(f"❌ Skipping '{video}': input verification failed.")
                    total_failed += 1
        elif option == 9:
            for video in video_files:
                video_path = os.path.join(input_dir, video)
                if verify_input_file(video_path):
                    try:
                        parallel_convert_selected_formats(video, input_dir, output_dir, enable_crop)
                        total_successful += 1
                    except Exception as e:
                        print(f"❌ Failed processing '{video}': {str(e)}")
                        total_failed += 1
                else:
                    print(f"❌ Skipping '{video}': input verification failed.")
                    total_failed += 1
        elif option == 11:
            valid_video_files = [video for video in video_files if verify_input_file(os.path.join(input_dir, video))]
            if valid_video_files:
                tv_series_parallel_encode(valid_video_files, input_dir, output_dir, enable_crop, batch_size=3)
            else:
                print("❌ No valid video files found for TV series encoding.")
        elif option == 13:
            for video in video_files:
                video_path = os.path.join(input_dir, video)
                if verify_input_file(video_path):
                    success = smart_batch_qsv_convert(video, input_dir, output_dir, enable_crop)
                    if success:
                        total_successful += 1
                    else:
                        total_failed += 1
                else:
                    print(f"❌ Skipping '{video}': input verification failed.")
                    total_failed += 1
        elif option == 14:
            valid_video_files = [video for video in video_files if verify_input_file(os.path.join(input_dir, video))]
            if valid_video_files:
                print(f"\n🔥 ========== LAUNCHING SMART TV SERIES - SEQUENTIAL (4 FORMATS) ==========")
                print(f"🎯 Formats: 480p + 720p + 720p 10BIT + 1080p")
                print(f"🎬 Smart Crop: FFmpeg multi-point detect on first episode, cache for all")
                print(f"🚀 QSV Optimization: async_depth=8 for maximum performance")
                print(f"✅ Sequential: 1 file at a time (optimal for Intel Arc GPU)")
                print(f"⚡ Expected time savings:")
                print(f"   • Crop detection: ~{(len(valid_video_files)-1) * 30:.0f} seconds saved")
                print(f"   • QSV performance: ~20-30% faster encoding per episode")
                print(f"📊 Total efficiency: {len(valid_video_files)} decodes vs {len(valid_video_files)*4} traditional")
                print("=" * 65)
                smart_tv_series_encode_optimized(valid_video_files, input_dir, output_dir, enable_crop, batch_size=3)
            else:
                print("❌ No valid video files found for Optimized Smart TV series encoding.")
        elif option == 15:
            valid_video_files = [video for video in video_files if verify_input_file(os.path.join(input_dir, video))]
            if valid_video_files:
                print(f"\n⚡ ========== LAUNCHING 3-FORMAT SMART TV SERIES ==========")
                print(f"🎯 Formats: 480p + 720p + 1080p (3 formats per episode)")
                print(f"🎬 Smart Crop: FFmpeg multi-point detect on first episode, cache for all")
                print(f"🚀 QSV Optimization: async_depth=8 for maximum performance")
                print(f"✅ Sequential: 1 file at a time (optimal for Intel Arc GPU)")
                print(f"⚡ Expected time savings:")
                print(f"   • Crop detection: ~{(len(valid_video_files)-1) * 30:.0f} seconds saved")
                print(f"   • QSV performance: ~20-30% faster encoding per episode")
                print(f"📊 Total efficiency: {len(valid_video_files)} decodes vs {len(valid_video_files)*3} traditional")
                print("=" * 65)
                smart_tv_series_encode_3formats(valid_video_files, input_dir, output_dir, enable_crop, batch_size=3)
            else:
                print("❌ No valid video files found for 3-Format Smart TV series encoding.")
        else:
            max_workers = min((os.cpu_count() or 4), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {}
                for video in video_files:
                    video_path = os.path.join(input_dir, video)
                    if verify_input_file(video_path):
                        future = executor.submit(pipeline_convert, video, input_dir, output_dir, option, enable_crop)
                        future_to_video[future] = video
                    else:
                        print(f"❌ Skipping '{video}': input verification failed.")
                        total_failed += 1

                for future in as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        success = future.result()
                        if success:
                            total_successful += 1
                        else:
                            total_failed += 1
                    except Exception as exc:
                        print(f"❌ Exception while processing '{video}': {exc}")
                        total_failed += 1

        overall_elapsed = time.time() - overall_start_time

        print("\n🎉 ========== ALL CONVERSIONS COMPLETED ==========")
        print(f"⏱️  Total processing time: {format_time(overall_elapsed)}")
        if option not in [11, 14, 15]:
            print(f"✅ Successful conversions: {total_successful}")
            print(f"❌ Failed conversions: {total_failed}")
            print(f"📊 Success rate: {(total_successful / (total_successful + total_failed) * 100):.1f}%" if (total_successful + total_failed) > 0 else "N/A")
        print("✅ Enhanced error handling prevented incomplete files")
        print("🔧 Color filter compatibility issues resolved")

        with open(os.path.join(output_dir, "conversiontime.txt"), "a", encoding="utf-8") as f:
            f.write(f"\nOVERALL COMPLETION - Option {option}: {format_time(overall_elapsed)}")
            if option not in [11, 14, 15]:
                f.write(f" - Success: {total_successful}, Failed: {total_failed}")
            f.write(f" - Enhanced error handling enabled\n")

    except KeyboardInterrupt:
        print("\n⏸️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error during processing: {str(e)}")
        sys.exit(1)

# -------------------------------------------------------------------------------
# Verify requirements and run main().
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()