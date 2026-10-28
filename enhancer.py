"""Làm nét ảnh chân dung nhưng giữ nguyên nét mặt.

Quy trình:
1. Real-ESRGAN phóng to + làm nét toàn ảnh (nền, quần áo, tóc...).
2. YuNet tìm khuôn mặt và 5 điểm mốc (2 mắt, mũi, 2 khóe miệng).
3. Mỗi khuôn mặt được căn thẳng về khung 512x512, đưa qua CodeFormer.
   - `fidelity` (0..1): càng cao càng giữ đúng nét mặt gốc, càng thấp AI càng "vẽ lại".
4. Trộn mặt đã phục hồi với mặt gốc (đã phóng to) theo `face_blend`,
   rồi dán ngược vào ảnh bằng mặt nạ mềm để không lộ đường viền.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import spandrel_extra_arches
from spandrel import ModelLoader

spandrel_extra_arches.install()  # CodeFormer nằm trong gói extra

MODELS = Path(__file__).parent / "models"

# 5 điểm mốc chuẩn của khuôn mặt trong khung 512x512 (FFHQ, dùng cho CodeFormer/GFPGAN)
FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],  # mắt trái (phía trái ảnh)
        [318.90277, 240.19360],  # mắt phải
        [256.63416, 314.01935],  # mũi
        [201.26117, 371.41043],  # khóe miệng trái
        [313.08905, 371.15118],  # khóe miệng phải
    ],
    dtype=np.float32,
)

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
DEVICE = torch.device("cpu")


def _img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def _tensor_to_img(t: torch.Tensor) -> np.ndarray:
    rgb = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


class Enhancer:
    def __init__(self):
        self._sr: dict[int, object] = {}
        self._face = None
        self._detector = None

    # ---------- nạp model (lười, chỉ khi cần) ----------
    def sr_model(self, scale: int):
        if scale not in self._sr:
            name = {2: "RealESRGAN_x2plus.pth", 4: "RealESRGAN_x4plus.pth"}[scale]
            m = ModelLoader(device=DEVICE).load_from_file(MODELS / name)
            m.model.eval()
            self._sr[scale] = m
        return self._sr[scale]

    def face_model(self):
        if self._face is None:
            m = ModelLoader(device=DEVICE).load_from_file(MODELS / "codeformer.pth")
            m.model.eval()
            self._face = m
        return self._face

    def detector(self, w: int, h: int):
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (w, h), 0.6, 0.3, 5000
            )
        self._detector.setInputSize((w, h))
        return self._detector

    # ---------- bước 1: phóng to toàn ảnh ----------
    @torch.inference_mode()
    def upscale(self, img: np.ndarray, scale: int, tile: int = 256, pad: int = 16) -> np.ndarray:
        if scale == 1:
            return img
        model = self.sr_model(scale)
        s = model.scale
        h, w = img.shape[:2]
        out = np.zeros((h * s, w * s, 3), np.uint8)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                # cắt ô có lề để tránh vệt nối giữa các ô
                y0, x0 = max(y - pad, 0), max(x - pad, 0)
                y1, x1 = min(y + tile + pad, h), min(x + tile + pad, w)
                # x2plus cần kích thước chẵn
                ph, pw = (y1 - y0) % 2, (x1 - x0) % 2
                patch = img[y0:y1, x0:x1]
                if ph or pw:
                    patch = cv2.copyMakeBorder(patch, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                res = _tensor_to_img(model(_img_to_tensor(patch)))
                ty, tx = (y - y0) * s, (x - x0) * s
                th, tw = (min(y + tile, h) - y) * s, (min(x + tile, w) - x) * s
                out[y * s : y * s + th, x * s : x * s + tw] = res[ty : ty + th, tx : tx + tw]
        return out

    # ---------- bước 2: tìm mặt ----------
    def detect_faces(self, img: np.ndarray) -> list[np.ndarray]:
        h, w = img.shape[:2]
        # ảnh nhỏ thì phóng to tạm để dò mặt tốt hơn
        k = 1.0 if max(h, w) >= 640 else 640 / max(h, w)
        probe = cv2.resize(img, None, fx=k, fy=k) if k != 1.0 else img
        _, faces = self.detector(probe.shape[1], probe.shape[0]).detect(probe)
        result = []
        if faces is None:
            return result
        for f in faces:
            pts = f[4:14].reshape(5, 2) / k
            eyes = pts[:2][np.argsort(pts[:2, 0])]
            mouth = pts[3:5][np.argsort(pts[3:5, 0])]
            result.append(np.vstack([eyes, pts[2:3], mouth]).astype(np.float32))
        return result

    # ---------- bước 3: phục hồi một khuôn mặt ----------
    @torch.inference_mode()
    def restore_face(self, crop: np.ndarray, fidelity: float) -> np.ndarray:
        model = self.face_model()
        net = model.model
        x = _img_to_tensor(crop)
        # CodeFormer: weight càng cao càng giữ đúng mặt gốc
        y = net(x * 2 - 1, weight=fidelity)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return _tensor_to_img((y + 1) / 2)

    # ---------- toàn bộ quy trình ----------
    def enhance(
        self,
        img: np.ndarray,
        scale: int = 2,
        fidelity: float = 0.8,
        face_blend: float = 0.85,
        restore_faces: bool = True,
        progress=lambda p, msg: None,
    ) -> tuple[np.ndarray, int]:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        progress(0.05, "Đang tìm khuôn mặt...")
        faces = self.detect_faces(img) if restore_faces else []

        progress(0.15, "Đang phóng to & làm nét toàn ảnh (Real-ESRGAN)...")
        up = self.upscale(img, scale)
        if not faces:
            return up, 0

        H, W = up.shape[:2]
        result = up.astype(np.float32)
        for i, lm in enumerate(faces):
            progress(0.6 + 0.35 * i / len(faces), f"Đang phục hồi khuôn mặt {i + 1}/{len(faces)} (CodeFormer)...")
            lm_up = lm * scale
            M, _ = cv2.estimateAffinePartial2D(lm_up, FACE_TEMPLATE, method=cv2.LMEDS)
            if M is None:
                continue
            crop = cv2.warpAffine(up, M, (512, 512), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            restored = self.restore_face(crop, fidelity)
            # trộn với mặt gốc để giữ nét đặc trưng (nốt ruồi, dáng mắt, mũi...)
            restored = cv2.addWeighted(restored, face_blend, crop, 1 - face_blend, 0)

            Minv = cv2.invertAffineTransform(M)
            back = cv2.warpAffine(restored, Minv, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

            # mặt nạ mềm: chỉ dán vùng mặt, viền mờ dần
            mask = np.zeros((512, 512), np.float32)
            cv2.rectangle(mask, (32, 32), (480, 480), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), 24)
            mask = cv2.warpAffine(mask, Minv, (W, H), flags=cv2.INTER_LINEAR)[..., None]
            result = result * (1 - mask) + back.astype(np.float32) * mask

        progress(1.0, "Xong")
        return np.clip(result, 0, 255).astype(np.uint8), len(faces)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Làm nét ảnh, giữ nét mặt")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--scale", type=int, default=2, choices=[1, 2, 4])
    ap.add_argument("--fidelity", type=float, default=0.8)
    ap.add_argument("--blend", type=float, default=0.85)
    a = ap.parse_args()

    src = cv2.imdecode(np.fromfile(a.input, np.uint8), cv2.IMREAD_UNCHANGED)
    out, n = Enhancer().enhance(src, a.scale, a.fidelity, a.blend, progress=lambda p, m: print(f"[{p:4.0%}] {m}"))
    cv2.imencode(Path(a.output).suffix or ".png", out)[1].tofile(a.output)
    print(f"Đã lưu {a.output} ({out.shape[1]}x{out.shape[0]}), {n} khuôn mặt")
