# Làm nét ảnh – giữ nguyên nét mặt

Công cụ làm nét và phóng to ảnh chân dung chạy trên máy của bạn, **không cần GPU**, **không gửi ảnh lên mạng**.
Khác với các app "làm đẹp", chương trình cố giữ đúng khuôn mặt gốc: dáng mắt, mũi, môi, nốt ruồi… không bị AI vẽ lại thành người khác.

*Sharpen and upscale portrait photos locally (CPU only) while keeping the person's facial identity. Real-ESRGAN + CodeFormer with a fidelity slider, Gradio web UI.*

## Cách hoạt động

1. **Real-ESRGAN** phóng to (1×, 2×, 4×) và làm nét toàn bộ ảnh: nền, tóc, quần áo.
2. **YuNet** (OpenCV) tìm các khuôn mặt và 5 điểm mốc (2 mắt, mũi, 2 khóe miệng).
3. Mỗi khuôn mặt được căn thẳng về khung 512×512 và phục hồi bằng **CodeFormer**.
4. Mặt đã phục hồi được trộn với mặt gốc rồi dán lại vào ảnh với viền mờ, không lộ vết ghép.

Hai thanh trượt quyết định mức "giống người thật":

| Thanh trượt | Mặc định | Ý nghĩa |
|---|---|---|
| **Độ trung thực khuôn mặt** | 0.8 | Cao = giữ đúng nét mặt gốc; thấp = AI tự vẽ thêm chi tiết (dễ khác người) |
| **Mức áp dụng AI cho mặt** | 0.85 | 0 = giữ nguyên mặt gốc (chỉ phóng to); 1 = dùng hoàn toàn mặt do AI tạo |

Nếu thấy mặt hơi "lạ", hãy tăng độ trung thực hoặc giảm mức áp dụng AI xuống khoảng 0.6–0.7.

## Cài đặt và chạy

Cần **Python 3.10 trở lên** ([tải tại đây](https://www.python.org/downloads/); khi cài trên Windows nhớ tick *Add Python to PATH*) và khoảng **2GB** dung lượng trống.

```bash
git clone https://github.com/conslmhoang/Enhance-image.git
cd Enhance-image
```

- **Windows:** nhấp đúp `chay.bat`.
- **macOS / Linux:** chạy `./chay.sh`.

Lần đầu chạy sẽ tự cài thư viện (~1GB) và tải model (~490MB), mất vài phút. Sau đó trình duyệt tự mở trang `http://127.0.0.1:7860`: kéo ảnh vào, bấm **Làm nét**, kéo thanh trước/sau để so sánh và tải ảnh PNG về. Ảnh kết quả cũng được lưu trong thư mục `ket_qua/`.

### Dùng bằng dòng lệnh

```bash
python enhancer.py anh_vao.jpg anh_ra.png --scale 2 --fidelity 0.8 --blend 0.85
```

## Tốc độ

Chạy trên CPU: ảnh khoảng 500×700 px, phóng 2× mất khoảng 30 giây. Phóng 4× chậm hơn nhiều.

## Giới hạn

- AI không tạo lại được chi tiết đã mất hẳn: chữ quá mờ vẫn không đọc được.
- Khuôn mặt quá nhỏ, nghiêng nhiều hoặc bị che khuất có thể không được nhận diện; khi đó ảnh vẫn được làm nét nhưng mặt không được phục hồi riêng.

## Giấy phép

- Mã nguồn trong repo này: [MIT](LICENSE).
- Model **không** nằm trong repo mà được tải từ nguồn gốc, mỗi model có giấy phép riêng:
  - [CodeFormer](https://github.com/sczhou/CodeFormer): S-Lab License 1.0, **chỉ dùng phi thương mại**.
  - [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): BSD-3-Clause.
  - [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet): MIT.

Vì CodeFormer chỉ cho phép dùng phi thương mại, **không dùng công cụ này cho mục đích kinh doanh**.

Chỉ dùng với ảnh bạn có quyền sử dụng, và tôn trọng quyền riêng tư của người trong ảnh.
