"""Giao diện web: chạy `python app.py` rồi mở http://127.0.0.1:7860.

Cùng file này cũng chạy được trên Hugging Face Spaces (tự nhận biết qua biến SPACE_ID).
"""
import os
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from download_models import ensure_models

# Trên Spaces: máy chủ miễn phí chỉ có 2 CPU nên giới hạn cỡ ảnh, và không lưu ảnh của người dùng
ON_SPACE = bool(os.environ.get("SPACE_ID"))
MAX_SIDE = 1024 if ON_SPACE else None
SCALES = [1, 2] if ON_SPACE else [1, 2, 4]

ensure_models()
from enhancer import Enhancer  # noqa: E402  (cần có model trước)

OUT_DIR = Path(tempfile.gettempdir()) / "lam_net_anh" if ON_SPACE else Path(__file__).parent / "ket_qua"
OUT_DIR.mkdir(parents=True, exist_ok=True)
enhancer = Enhancer()


def run(image, scale, fidelity, blend, restore_faces, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Hãy chọn một ảnh trước.")
    note = ""
    if MAX_SIDE and max(image.shape[:2]) > MAX_SIDE:
        k = MAX_SIDE / max(image.shape[:2])
        image = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        note = f" · ảnh lớn đã được thu nhỏ về {image.shape[1]}×{image.shape[0]} trước khi xử lý"
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    t0 = time.time()
    out, n = enhancer.enhance(
        bgr,
        scale=int(scale),
        fidelity=fidelity,
        face_blend=blend,
        restore_faces=restore_faces,
        progress=lambda p, m: progress(p, desc=m),
    )
    path = OUT_DIR / f"lam_net_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imencode(".png", out)[1].tofile(str(path))
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # ảnh gốc phóng cùng cỡ để so sánh trước/sau
    before = cv2.resize(image, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    info = (
        f"Xong trong {time.time() - t0:.1f}s · {rgb.shape[1]}×{rgb.shape[0]} px · "
        f"{n} khuôn mặt được phục hồi" + ("" if ON_SPACE else f" · đã lưu: {path.name}") + note
    )
    return (before, rgb), str(path), info


# xoá file tạm sau 1 giờ
with gr.Blocks(title="Làm nét ảnh", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(
        "## Làm nét ảnh – giữ nguyên nét mặt\n"
        "Real-ESRGAN làm nét toàn ảnh, CodeFormer phục hồi khuôn mặt. "
        "Nếu mặt bị \"khác người\", hãy **tăng Độ trung thực** hoặc **giảm Mức áp dụng AI cho mặt**."
        + (
            "\n\nBản chạy trên máy chủ miễn phí (2 CPU): mỗi ảnh mất khoảng 1–2 phút, ảnh lớn được thu nhỏ còn "
            f"tối đa {MAX_SIDE}px. Ảnh tải lên và kết quả tự xoá sau 1 giờ. "
            "Muốn nhanh hơn và phóng 4×, hãy [chạy trên máy mình](https://github.com/conslmhoang/Enhance-image)."
            if ON_SPACE
            else ""
        )
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Ảnh gốc", type="numpy")
            scale = gr.Radio(SCALES, value=2, label="Phóng to (lần)")
            fidelity = gr.Slider(0, 1, value=0.8, step=0.05, label="Độ trung thực khuôn mặt",
                                 info="Cao = giữ đúng nét mặt gốc; thấp = AI tự vẽ thêm chi tiết")
            blend = gr.Slider(0, 1, value=0.85, step=0.05, label="Mức áp dụng AI cho mặt",
                              info="0 = giữ mặt gốc (chỉ phóng to); 1 = dùng hoàn toàn mặt AI")
            faces = gr.Checkbox(value=True, label="Phục hồi khuôn mặt")
            btn = gr.Button("Làm nét", variant="primary")
        with gr.Column(scale=2):
            cmp = gr.ImageSlider(label="Trước / Sau (kéo thanh để so sánh)", type="numpy")
            info = gr.Markdown()
            file = gr.File(label="Tải ảnh kết quả (PNG)")
    btn.click(run, [inp, scale, fidelity, blend, faces], [cmp, file, info])

if __name__ == "__main__":
    if ON_SPACE:
        demo.queue(max_size=20).launch()
    else:
        demo.queue().launch(server_name="127.0.0.1", inbrowser=True)
