#!/usr/bin/env python3
"""
Video Segmentation Tool
========================
Splits large video files into smaller chunks of 200MB each.
Useful for uploading videos to the ARAS Auto-Annotation Studio when file size exceeds 200MB.

Usage:
    python segment_video.py <input_video> [output_directory] [chunk_size_mb]

Examples:
    python segment_video.py large_video.mp4
    python segment_video.py large_video.mp4 ./segmented_videos 200
    python segment_video.py /path/to/video.mp4 ./output 150

Output:
    Creates numbered segments: video_segment_0.mp4, video_segment_1.mp4, etc.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"❌ Error getting video duration: {e}")
        return None


def get_video_bitrate(video_path: str) -> Optional[int]:
    """Get video bitrate in bits/second using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        bitrate_str = result.stdout.strip()
        if bitrate_str and bitrate_str != "N/A":
            return int(bitrate_str)
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def calculate_segment_duration(
    video_path: str, chunk_size_mb: int, total_duration: float
) -> float:
    """Calculate how long each segment should be based on target chunk size."""
    file_size_mb = get_file_size_mb(video_path)
    if file_size_mb <= 0:
        return 0
    bytes_per_second = (file_size_mb * 1024 * 1024) / total_duration
    segment_duration = (chunk_size_mb * 1024 * 1024) / bytes_per_second
    return segment_duration


def segment_video(
    input_video: str,
    output_dir: str = "./segmented_videos",
    chunk_size_mb: int = 200,
    codec: str = "copy",
) -> bool:
    """
    Segment a video file into chunks of specified size.

    Args:
        input_video: Path to input video file
        output_dir: Directory to save segmented videos
        chunk_size_mb: Target size for each segment in MB
        codec: Video codec to use ('copy' for no re-encoding, or 'libx264', etc.)

    Returns:
        bool: True if successful, False otherwise
    """
    # Validate input
    if not os.path.exists(input_video):
        print(f"❌ Error: Input video '{input_video}' not found")
        return False

    input_path = Path(input_video)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get video info
    print(f"\n📹 Analyzing video: {input_path.name}")
    print(f"   Location: {input_path.absolute()}")

    file_size_mb = get_file_size_mb(input_video)
    print(f"   File size: {file_size_mb:.2f} MB")

    # Check if segmentation is needed
    if file_size_mb <= chunk_size_mb:
        print(f"\n✓ Video is already under {chunk_size_mb}MB ({file_size_mb:.2f}MB)")
        print("  No segmentation needed!")
        return True

    total_duration = get_video_duration(input_video)
    if total_duration is None:
        return False

    print(f"   Duration: {total_duration / 60:.2f} minutes")

    # Calculate segment duration
    segment_duration = calculate_segment_duration(input_video, chunk_size_mb, total_duration)

    if segment_duration <= 0:
        print("❌ Error: Could not calculate segment duration")
        return False

    num_segments = int(total_duration / segment_duration) + 1

    print(f"\n⚙️  Segmentation Plan:")
    print(f"   Target chunk size: {chunk_size_mb}MB")
    print(f"   Estimated segments: {num_segments}")
    print(f"   Segment duration: {segment_duration / 60:.2f} minutes each")
    print(f"\n📁 Output directory: {output_path.absolute()}")
    print(f"\n🎬 Starting segmentation...\n")

    # Segment the video
    segment_num = 0
    start_time = 0

    while start_time < total_duration:
        end_time = min(start_time + segment_duration, total_duration)
        output_file = output_path / f"{input_path.stem}_segment_{segment_num:03d}.mp4"

        duration = end_time - start_time

        print(f"[{segment_num + 1}/{num_segments}] Creating segment: {output_file.name}")
        print(f"          Time range: {start_time / 60:.2f}m - {end_time / 60:.2f}m ({duration / 60:.2f}m)")

        cmd = [
            "ffmpeg",
            "-i",
            input_video,
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-c:v",
            codec,
            "-c:a",
            "aac",
            "-y",  # Overwrite output file
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            output_size = get_file_size_mb(str(output_file))
            print(f"          ✓ Created ({output_size:.2f}MB)\n")
            segment_num += 1
            start_time = end_time
        except subprocess.CalledProcessError as e:
            print(f"          ❌ Error creating segment: {e}\n")
            return False

    print(f"\n✅ Segmentation completed successfully!")
    print(f"   Total segments created: {segment_num}")
    print(f"   Output directory: {output_path.absolute()}\n")

    # Print instructions
    print("📋 Next Steps:")
    print(f"   1. Upload each segment file separately to ARAS Auto-Annotation Studio")
    print(f"   2. All segments will be saved to the same output_frames directory")
    print(f"   3. All segments can be processed together for unified annotation\n")

    return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./segmented_videos"
    chunk_size_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    success = segment_video(input_video, output_dir, chunk_size_mb)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
