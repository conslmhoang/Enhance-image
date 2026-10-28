"""Giao diện web cục bộ: chạy `python app.py` rồi mở http://127.0.0.1:7860"""
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from enhancer import Enhancer

OUT_DIR = Path(__file__).parent / "ket_qua"
OUT_DIR.mkdir(exist_ok=True)
enhancer = Enhancer()


def run(image, scale, fidelity, blend, restore_faces, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Hãy chọn một ảnh trước.")
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
        f"{n} khuôn mặt được phục hồi · đã lưu: {path.name}"
    )
    return (before, rgb), str(path), info


with gr.Blocks(title="Làm nét ảnh") as demo:
    gr.Markdown(
        "## Làm nét ảnh – giữ nguyên nét mặt\n"
        "Real-ESRGAN làm nét toàn ảnh, CodeFormer phục hồi khuôn mặt. "
        "Nếu mặt bị \"khác người\", hãy **tăng Độ trung thực** hoặc **giảm Mức áp dụng AI cho mặt**."
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Ảnh gốc", type="numpy")
            scale = gr.Radio([1, 2, 4], value=2, label="Phóng to (lần)")
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
    demo.queue().launch(server_name="127.0.0.1", inbrowser=True)
