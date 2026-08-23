"""Download a free pig farm video from Pexels for training."""
import urllib.request
import json
import sys


def main():
    url = "https://api.pexels.com/videos/search?query=pig+farm+pen&per_page=10&orientation=landscape"
    req = urllib.request.Request(url, headers={
        "Authorization": "cbcLBrwO4YDsHnFOj9EhRY6HUqZdaZVjHIpXXxGkpkGSHiTI6zVL7VQD"
    })

    print("Querying Pexels API for free pig farm videos...")
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"Pexels API query failed: {e}")
        print("Falling back to Pixabay...")
        download_from_pixabay()
        return

    videos = data.get("videos", [])
    if not videos:
        print("No pig videos found on Pexels. Trying Pixabay...")
        download_from_pixabay()
        return

    print(f"Found {len(videos)} pig videos on Pexels:\n")
    for v in videos:
        vid_id = v["id"]
        duration = v["duration"]
        # Find a good quality file (SD or HD, 720p or lower for fast download)
        files = v.get("video_files", [])
        good_files = [f for f in files if f.get("height", 9999) <= 720 and f.get("width", 0) > 0]
        if not good_files:
            good_files = sorted(files, key=lambda f: f.get("height", 9999))
        if good_files:
            best = good_files[0]
            print(f"  ID: {vid_id} | Duration: {duration}s | {best.get('width')}x{best.get('height')} | Quality: {best.get('quality')}")

    # Pick the longest video for best training data
    best_video = max(videos, key=lambda v: v["duration"])
    vid_files = best_video.get("video_files", [])
    # Prefer 720p or closest
    candidates = sorted(vid_files, key=lambda f: abs(f.get("height", 9999) - 720))
    download_url = candidates[0]["link"] if candidates else None

    if not download_url:
        print("No downloadable file found.")
        return

    output_path = "data/raw/videos/pig_farm_pen.mp4"
    print(f"\nDownloading best video (ID: {best_video['id']}, {best_video['duration']}s)...")
    print(f"  URL: {download_url}")
    print(f"  Saving to: {output_path}")

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    urllib.request.urlretrieve(download_url, output_path)

    file_size = os.path.getsize(output_path)
    print(f"\n✅ Download complete! File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"   Saved to: {output_path}")


def download_from_pixabay():
    """Fallback: download from Pixabay API."""
    url = "https://pixabay.com/api/videos/?key=46498122-e98ecdf4f7d69c4c7cc4e9474&q=pig+farm&per_page=5"
    req = urllib.request.Request(url)

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"Pixabay API also failed: {e}")
        print("\nPlease manually download a pig video from:")
        print("  https://www.pexels.com/search/videos/pig/")
        print("  https://pixabay.com/videos/search/pig/")
        print("\nThen place it at: data/raw/videos/pig_farm_pen.mp4")
        return

    hits = data.get("hits", [])
    if not hits:
        print("No videos found on Pixabay either.")
        return

    best = max(hits, key=lambda h: h.get("duration", 0))
    vid_url = best.get("videos", {}).get("medium", {}).get("url")
    if not vid_url:
        vid_url = best.get("videos", {}).get("small", {}).get("url")

    if vid_url:
        output_path = "data/raw/videos/pig_farm_pen.mp4"
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Downloading from Pixabay (duration: {best.get('duration')}s)...")
        urllib.request.urlretrieve(vid_url, output_path)
        file_size = os.path.getsize(output_path)
        print(f"\n✅ Download complete! File size: {file_size / 1024 / 1024:.1f} MB")
        print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    main()
