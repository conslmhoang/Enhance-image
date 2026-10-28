"""Tải các model cần thiết vào thư mục models/ (khoảng 490MB, chỉ cần chạy một lần)."""
import sys
import time
import urllib.request
from pathlib import Path

MODELS = Path(__file__).parent / "models"

FILES = {
    "RealESRGAN_x2plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        67061725,
    ),
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        67040989,
    ),
    "codeformer.pth": (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        376637898,
    ),
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        232589,
    ),
}


def download(name: str, url: str, size: int, retries: int = 8) -> None:
    dst = MODELS / name
    for attempt in range(1, retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if have == size:
            print(f"  ✓ {name}")
            return
        if have > size:
            dst.unlink()
            have = 0
        try:
            # tải tiếp từ chỗ bị đứt (mạng chập chờn hay làm hỏng file lớn)
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "ab" if have else "wb") as f:
                if have and r.status != 206:  # máy chủ không hỗ trợ tải tiếp
                    f.truncate(0)
                    have = 0
                done = have
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    print(f"\r  ↓ {name}: {done / 1e6:6.1f}/{size / 1e6:.1f} MB", end="", flush=True)
            print()
        except Exception as e:
            print(f"\n  ! {name}: lỗi lần {attempt} ({e}), thử lại...")
            time.sleep(2)
    if not dst.exists() or dst.stat().st_size != size:
        sys.exit(f"Không tải được {name}. Hãy chạy lại lệnh này.")
    print(f"  ✓ {name}")


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    print("Đang kiểm tra / tải model...")
    for name, (url, size) in FILES.items():
        download(name, url, size)
    print("Đã đủ model.")
