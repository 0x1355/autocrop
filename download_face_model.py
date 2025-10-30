#!/usr/bin/env python
"""
Download DSFD face detection model from HuggingFace mirror
since the original URL (folk.ntnu.no) is no longer available.
"""
import os
import urllib.request
import sys
from pathlib import Path

# HuggingFace mirror URL
HUGGINGFACE_URL = "https://huggingface.co/zixianma/mma/resolve/main/WIDERFace_DSFD_RES152.pth"

# PyTorch cache directory
CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
MODEL_PATH = CACHE_DIR / "WIDERFace_DSFD_RES152.pth"

def download_with_progress(url, destination):
    """Download file with progress bar"""
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}% [{count * block_size} / {total_size} bytes]")
        sys.stdout.flush()

    print(f"Downloading from: {url}")
    print(f"Saving to: {destination}")
    urllib.request.urlretrieve(url, destination, reporthook)
    print("\n✓ Download complete!")

def main():
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"✓ Model already exists at: {MODEL_PATH}")
        file_size = MODEL_PATH.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"  Size: {file_size:.2f} MB")
        return 0

    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Download the model
        download_with_progress(HUGGINGFACE_URL, MODEL_PATH)

        # Verify the file exists and get size
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / (1024 * 1024)  # Convert to MB
            print(f"✓ Model saved successfully!")
            print(f"  Location: {MODEL_PATH}")
            print(f"  Size: {file_size:.2f} MB")
            return 0
        else:
            print("✗ Download completed but file not found!")
            return 1

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nYou can manually download the model from:")
        print(f"  {HUGGINGFACE_URL}")
        print(f"\nAnd place it at:")
        print(f"  {MODEL_PATH}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
