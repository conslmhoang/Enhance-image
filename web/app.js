// Làm nét ảnh ngay trong trình duyệt: Real-ESRGAN (x2) + CodeFormer, tìm mặt bằng YuNet.
// Cùng quy trình với enhancer.py, chỉ khác là chạy bằng onnxruntime-web.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

const MODEL_CACHE = "enhance-models-v2"; // đổi tên khi đổi file model, bản cũ sẽ bị xoá
if (self.caches) caches.keys().then((ks) => ks.filter((k) => k.startsWith("enhance-models-") && k !== MODEL_CACHE).forEach((k) => caches.delete(k)));
const SR_URLS = ["web/models/realesrgan_x2.onnx"];
const FACE_URLS = ["web/models/codeformer.onnx.part0", "web/models/codeformer.onnx.part1", "web/models/codeformer.onnx.part2", "web/models/codeformer.onnx.part3", "web/models/codeformer.onnx.part4", "web/models/codeformer.onnx.part5", "web/models/codeformer.onnx.part6", "web/models/codeformer.onnx.part7"];
const DET_URLS = ["web/models/yunet.onnx"];
// kích thước thật của từng file: GitHub Pages nén gzip nên content-length không dùng để tính tiến độ được
const SIZES = {
  "web/models/realesrgan_x2.onnx": 33811958,
  "web/models/codeformer.onnx.part0": 24000000,
  "web/models/codeformer.onnx.part1": 24000000,
  "web/models/codeformer.onnx.part2": 24000000,
  "web/models/codeformer.onnx.part3": 24000000,
  "web/models/codeformer.onnx.part4": 24000000,
  "web/models/codeformer.onnx.part5": 24000000,
  "web/models/codeformer.onnx.part6": 24000000,
  "web/models/codeformer.onnx.part7": 20652126,
  "web/models/yunet.onnx": 232589,
};
const MAX_SIDE = 1280; // ảnh lớn hơn được thu nhỏ trước khi xử lý (trình duyệt có giới hạn bộ nhớ)
const TILE = 192, PAD = 8; // đều là số chẵn vì model x2 cần kích thước chẵn

// 5 điểm mốc chuẩn trong khung 512x512 (FFHQ) – giống enhancer.py
const TEMPLATE = [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]];

const $ = (id) => document.getElementById(id);
const statusEl = $("status"), prog = $("prog"), goBtn = $("go");
let srcImage = null;
let srSession = null, faceSession = null, detSession = null;
const useGPU = !!navigator.gpu;
let backend = "CPU"; // thiết bị thực sự dùng, hiện ở dòng trạng thái

function setStatus(msg, p) {
  statusEl.textContent = msg;
  if (p == null) prog.hidden = true;
  else { prog.hidden = false; prog.value = p; }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// ---------- tải model (lưu vào Cache Storage, lần sau không phải tải lại) ----------
async function fetchCached(urls, label) {
  const cache = self.caches ? await caches.open(MODEL_CACHE).catch(() => null) : null;
  const parts = [];
  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];
    let res = cache && (await cache.match(url));
    if (!res) {
      const chunks = [];
      let got = 0;
      const total = SIZES[url] || 0;
      // mạng chập chờn hay làm đứt file lớn: tự thử lại (tải lại từ đầu, vì file được nén gzip khi truyền)
      for (let attempt = 1; ; attempt++) {
        chunks.length = 0;
        got = 0;
        try {
          const r = await fetch(url);
          if (!r.ok) throw new Error(`Không tải được ${url} (${r.status})`);
          const reader = r.body.getReader();
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            got += value.length;
            const mb = (got / 1e6).toFixed(0);
            setStatus(`Đang tải model ${label} (${i + 1}/${urls.length}): ${mb}${total ? "/" + (total / 1e6).toFixed(0) : ""} MB`, total ? Math.min(1, got / total) : null);
          }
          if (total && got !== total) throw new Error(`File ${url} bị thiếu (${got}/${total} byte)`);
          break;
        } catch (e) {
          if (attempt >= 6) throw e;
          setStatus(`Mạng bị gián đoạn, đang thử tải lại (${attempt}/5)...`);
          await new Promise((r) => setTimeout(r, 1500 * attempt));
        }
      }
      const blob = new Blob(chunks);
      if (cache) await cache.put(url, new Response(blob)).catch(() => {});
      parts.push(new Uint8Array(await blob.arrayBuffer()));
    } else {
      parts.push(new Uint8Array(await res.arrayBuffer()));
    }
  }
  const size = parts.reduce((s, p) => s + p.length, 0);
  const all = new Uint8Array(size);
  let off = 0;
  for (const p of parts) { all.set(p, off); off += p.length; }
  return all;
}

async function createSession(bytes) {
  if (useGPU) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
      backend = "WebGPU";
      return s;
    }
    catch (e) { console.warn("WebGPU không dùng được, chuyển sang CPU:", e); }
  }
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

async function loadModels(needFace) {
  if (!srSession) {
    const b = await fetchCached(SR_URLS, "làm nét");
    setStatus("Đang khởi tạo model làm nét...");
    srSession = await createSession(b);
  }
  if (needFace && !faceSession) {
    const b = await fetchCached(FACE_URLS, "khuôn mặt");
    setStatus("Đang khởi tạo model khuôn mặt...");
    faceSession = await createSession(b);
  }
  if (needFace && !detSession) {
    const b = await fetchCached(DET_URLS, "tìm mặt");
    detSession = await ort.InferenceSession.create(b, { executionProviders: ["wasm"] });
  }
}

// ---------- tiện ích ảnh ----------
function canvasOf(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}
function toCHW(rgba, w, h, scale = 1, bias = 0) {
  const n = w * h, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[i * 4] / 255 * scale + bias;
    out[n + i] = rgba[i * 4 + 1] / 255 * scale + bias;
    out[2 * n + i] = rgba[i * 4 + 2] / 255 * scale + bias;
  }
  return out;
}

// ---------- bước 1: phóng to x2 theo từng ô ----------
async function upscale(srcCanvas, onProgress) {
  const w = srcCanvas.width, h = srcCanvas.height;
  const src = srcCanvas.getContext("2d").getImageData(0, 0, w, h).data;
  const W = w * 2, H = h * 2;
  const out = new ImageData(W, H);
  const tiles = Math.ceil(w / TILE) * Math.ceil(h / TILE);
  let done = 0;
  for (let y = 0; y < h; y += TILE) {
    for (let x = 0; x < w; x += TILE) {
      const x0 = Math.max(x - PAD, 0), y0 = Math.max(y - PAD, 0);
      const x1 = Math.min(x + TILE + PAD, w), y1 = Math.min(y + TILE + PAD, h);
      const pw = x1 - x0, ph = y1 - y0;
      const t = new Float32Array(3 * pw * ph), n = pw * ph;
      for (let yy = 0; yy < ph; yy++) {
        for (let xx = 0; xx < pw; xx++) {
          const si = ((y0 + yy) * w + (x0 + xx)) * 4, di = yy * pw + xx;
          t[di] = src[si] / 255; t[n + di] = src[si + 1] / 255; t[2 * n + di] = src[si + 2] / 255;
        }
      }
      const res = await srSession.run({ input: new ort.Tensor("float32", t, [1, 3, ph, pw]) });
      const o = res.output.data, OW = pw * 2, ON = OW * ph * 2;
      // chỉ chép phần lõi của ô (bỏ lề) để không có vệt nối
      const cx = (x - x0) * 2, cy = (y - y0) * 2;
      const cw = (Math.min(x + TILE, w) - x) * 2, chh = (Math.min(y + TILE, h) - y) * 2;
      for (let yy = 0; yy < chh; yy++) {
        for (let xx = 0; xx < cw; xx++) {
          const si = (cy + yy) * OW + (cx + xx), di = ((y * 2 + yy) * W + (x * 2 + xx)) * 4;
          out.data[di] = o[si] * 255; out.data[di + 1] = o[ON + si] * 255; out.data[di + 2] = o[2 * ON + si] * 255;
          out.data[di + 3] = 255;
        }
      }
      onProgress(++done / tiles);
      await tick();
    }
  }
  const c = canvasOf(W, H);
  c.getContext("2d").putImageData(out, 0, 0);
  return c;
}

// ---------- bước 2: tìm mặt (YuNet, khung vào cố định 640x640), lấy 5 điểm mốc ----------
async function detectFaces(canvas, scoreThr = 0.6, nmsThr = 0.3) {
  const S = 640, k = S / Math.max(canvas.width, canvas.height);
  const c = canvasOf(S, S), ctx = c.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, S, S);
  ctx.drawImage(canvas, 0, 0, Math.round(canvas.width * k), Math.round(canvas.height * k));
  const d = ctx.getImageData(0, 0, S, S).data, N = S * S, t = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) { t[i] = d[i * 4 + 2]; t[N + i] = d[i * 4 + 1]; t[2 * N + i] = d[i * 4]; } // BGR, 0..255
  const o = await detSession.run({ input: new ort.Tensor("float32", t, [1, 3, S, S]) });

  const cand = [];
  for (const st of [8, 16, 32]) {
    const cls = o[`cls_${st}`].data, obj = o[`obj_${st}`].data, bb = o[`bbox_${st}`].data, kp = o[`kps_${st}`].data;
    const cols = S / st;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
      if (score < scoreThr) continue;
      const r = Math.floor(i / cols), col = i % cols;
      const cx = (col + bb[i * 4]) * st, cy = (r + bb[i * 4 + 1]) * st;
      const w = Math.exp(bb[i * 4 + 2]) * st, h = Math.exp(bb[i * 4 + 3]) * st;
      const pts = [];
      for (let n = 0; n < 5; n++) pts.push([((kp[i * 10 + 2 * n] + col) * st) / k, ((kp[i * 10 + 2 * n + 1] + r) * st) / k]);
      cand.push({ score, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], pts });
    }
  }
  // loại các khung trùng nhau (NMS)
  cand.sort((a, b) => b.score - a.score);
  const iou = (a, b) => {
    const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = w * h;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
  };
  const keep = [];
  for (const f of cand) if (keep.every((g) => iou(f.box, g.box) < nmsThr)) keep.push(f);
  return keep.map(({ pts }) => {
    const eyes = pts.slice(0, 2).sort((a, b) => a[0] - b[0]);
    const mouth = pts.slice(3, 5).sort((a, b) => a[0] - b[0]);
    return [eyes[0], eyes[1], pts[2], mouth[0], mouth[1]];
  });
}

// phép biến đổi đồng dạng (xoay + tỉ lệ + dịch) khớp các điểm src -> dst theo bình phương tối thiểu
function similarity(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const px = src[i][0] - sx, py = src[i][1] - sy, qx = dst[i][0] - dx, qy = dst[i][1] - dy;
    num1 += px * qx + py * qy; num2 += px * qy - py * qx; den += px * px + py * py;
  }
  const a = num1 / den, b = num2 / den;
  return { a, b, tx: dx - (a * sx - b * sy), ty: dy - (b * sx + a * sy) };
}

// mặt nạ mềm 512x512: 1 ở giữa, mờ dần ra viền
const MASK = (() => {
  const m = new Float32Array(512 * 512);
  for (let y = 0; y < 512; y++) for (let x = 0; x < 512; x++) {
    const d = Math.min(x, y, 511 - x, 511 - y);
    const t = Math.min(1, Math.max(0, (d - 16) / 56));
    m[y * 512 + x] = t * t * (3 - 2 * t);
  }
  return m;
})();

// ---------- bước 3: phục hồi từng khuôn mặt rồi dán lại ----------
async function restoreFace(up, pts, fidelity, blend) {
  const M = similarity(pts, TEMPLATE);
  const crop = canvasOf(512, 512), cctx = crop.getContext("2d");
  cctx.imageSmoothingQuality = "high";
  cctx.setTransform(M.a, M.b, -M.b, M.a, M.tx, M.ty);
  cctx.drawImage(up, 0, 0);
  const cd = cctx.getImageData(0, 0, 512, 512);

  const res = await faceSession.run({
    input: new ort.Tensor("float32", toCHW(cd.data, 512, 512, 2, -1), [1, 3, 512, 512]),
    weight: new ort.Tensor("float32", new Float32Array([fidelity]), []),
  });
  const o = res.output.data, N = 512 * 512;
  const face = new ImageData(512, 512);
  for (let i = 0; i < N; i++) {
    for (let c = 0; c < 3; c++) {
      const ai = (Math.min(1, Math.max(-1, o[c * N + i])) + 1) * 127.5;
      // trộn với mặt gốc để giữ nét đặc trưng
      face.data[i * 4 + c] = ai * blend + cd.data[i * 4 + c] * (1 - blend);
    }
    face.data[i * 4 + 3] = MASK[i] * (cd.data[i * 4 + 3] / 255) * 255;
  }
  const fc = canvasOf(512, 512);
  fc.getContext("2d").putImageData(face, 0, 0);

  // dán ngược bằng phép biến đổi nghịch đảo
  const s2 = M.a * M.a + M.b * M.b, ia = M.a / s2, ib = -M.b / s2;
  const ex = -(ia * M.tx - ib * M.ty), fy = -(ib * M.tx + ia * M.ty);
  const uctx = up.getContext("2d");
  uctx.save();
  uctx.imageSmoothingQuality = "high";
  uctx.setTransform(ia, ib, -ib, ia, ex, fy);
  uctx.drawImage(fc, 0, 0);
  uctx.restore();
}

// ---------- giao diện ----------
function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    srcImage = img;
    $("thumb").src = url; $("thumb").hidden = false;
    $("dropText").textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight}. Bấm để đổi ảnh khác.`;
    goBtn.disabled = false;
    setStatus("Sẵn sàng. Bấm Làm nét.");
  };
  img.onerror = () => setStatus("Không đọc được ảnh này.");
  img.src = url;
}

const drop = $("drop"), fileInput = $("file");
fileInput.onchange = () => {
  loadFile(fileInput.files[0]);
  fileInput.value = ""; // để chọn lại cùng một ảnh vẫn nhận
};
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); loadFile(e.dataTransfer.files[0]); };
document.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (f) loadFile(f);
});
for (const id of ["fid", "blend"]) $(id).oninput = () => ($(id + "V").textContent = $(id).value);

function showResult(before, after, info) {
  const box = $("result");
  box.className = "cmp";
  box.innerHTML = "";
  after.style.display = "block";
  before.classList.add("before");
  const bar = document.createElement("div"); bar.className = "bar";
  const t1 = document.createElement("span"); t1.className = "tag"; t1.style.left = "8px"; t1.textContent = "Trước";
  const t2 = document.createElement("span"); t2.className = "tag"; t2.style.right = "8px"; t2.textContent = "Sau";
  box.append(after, before, bar, t1, t2);
  const set = (clientX) => {
    const r = box.getBoundingClientRect();
    const p = Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100));
    before.style.clipPath = `inset(0 ${100 - p}% 0 0)`;
    bar.style.left = p + "%";
  };
  let drag = false;
  box.onpointerdown = (e) => { drag = true; box.setPointerCapture(e.pointerId); set(e.clientX); };
  box.onpointermove = (e) => drag && set(e.clientX);
  box.onpointerup = () => (drag = false);
  $("info").textContent = info;
}

goBtn.onclick = async () => {
  if (!srcImage) return;
  goBtn.disabled = true;
  const scale = +document.querySelector("input[name=scale]:checked").value;
  const fidelity = +$("fid").value, blend = +$("blend").value, doFaces = $("faces").checked;
  try {
    const t0 = performance.now();
    await loadModels(doFaces);

    // ảnh vào: thu nhỏ nếu quá lớn, làm tròn kích thước về số chẵn
    let w = srcImage.naturalWidth, h = srcImage.naturalHeight, note = "";
    const k = Math.min(1, MAX_SIDE / Math.max(w, h));
    if (k < 1) note = ` · ảnh lớn đã được thu nhỏ về ${Math.round(w * k)}×${Math.round(h * k)}`;
    w = Math.max(2, Math.round(w * k / 2) * 2); h = Math.max(2, Math.round(h * k / 2) * 2);
    const src = canvasOf(w, h), sctx = src.getContext("2d");
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(srcImage, 0, 0, w, h);

    let faces = [];
    if (doFaces) { setStatus("Đang tìm khuôn mặt..."); await tick(); faces = await detectFaces(src); }

    const up = await upscale(src, (p) => setStatus("Đang làm nét toàn ảnh (Real-ESRGAN)...", p));
    for (let i = 0; i < faces.length; i++) {
      setStatus(`Đang phục hồi khuôn mặt ${i + 1}/${faces.length} (CodeFormer)...`, i / faces.length);
      await tick();
      await restoreFace(up, faces[i].map(([x, y]) => [x * 2, y * 2]), fidelity, blend);
    }

    // 1× = làm nét ở cỡ x2 rồi thu về cỡ gốc
    let out = up;
    if (scale === 1) {
      out = canvasOf(w, h);
      const octx = out.getContext("2d");
      octx.imageSmoothingQuality = "high";
      octx.drawImage(up, 0, 0, w, h);
    }
    const before = canvasOf(out.width, out.height), bctx = before.getContext("2d");
    bctx.imageSmoothingQuality = "high";
    bctx.drawImage(srcImage, 0, 0, out.width, out.height);

    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    showResult(before, out, `Xong trong ${secs}s · ${out.width}×${out.height} px · ${faces.length} khuôn mặt được phục hồi${note}`);
    out.toBlob((blob) => {
      const a = $("dl");
      if (a.href) URL.revokeObjectURL(a.href);
      a.href = URL.createObjectURL(blob);
      a.download = "lam_net.png";
      a.hidden = false;
    }, "image/png");
    setStatus(`Xong (chạy bằng ${backend}).`);
  } catch (e) {
    console.error(e);
    setStatus("Lỗi: " + (e.message || e));
  } finally {
    goBtn.disabled = false;
  }
};

if (!self.caches) console.warn("Trình duyệt không hỗ trợ Cache Storage: model sẽ được tải lại mỗi lần.");
