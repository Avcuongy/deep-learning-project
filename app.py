from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import html
from pathlib import Path
from typing import Iterable, cast

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model.lstm import ImageCaptioningModel as LSTMImageCaptioningModel
from model.transformer import ImageCaptioningModel as TransformerImageCaptioningModel

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "mimic_train.csv"
DEFAULT_LSTM_CKPT = PROJECT_ROOT / "models" / "model1.pth"
DEFAULT_TRANSFORMER_CKPT = PROJECT_ROOT / "models" / "model2.pth"

IMAGE_SIZE = 224
IMAGE_MEAN = 0.4888
IMAGE_STD = 0.2839
FREQ_THRESHOLD = 10
DEFAULT_MAX_LEN = 100


@dataclass(frozen=True)
class ModelSpec:
    label: str
    architecture: str
    checkpoint: Path
    embed_size: int
    num_layers: int
    num_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    max_len: int = 512


MODEL_SPECS = {
    "LSTM": ModelSpec(
        label="LSTM",
        architecture="lstm",
        checkpoint=DEFAULT_LSTM_CKPT,
        embed_size=256,
        num_layers=4,
        dropout=0.2,
    ),
    "Transformer": ModelSpec(
        label="Transformer",
        architecture="transformer",
        checkpoint=DEFAULT_TRANSFORMER_CKPT,
        embed_size=256,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        dropout=0.3,
    ),
}


class Vocabulary:
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, freq_threshold: int = 10):
        self.freq_threshold = freq_threshold
        self.itos = {
            self.PAD: "<PAD>",
            self.SOS: "<SOS>",
            self.EOS: "<EOS>",
            self.UNK: "<UNK>",
        }
        self.stoi = {token: index for index, token in self.itos.items()}

    def __len__(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> list[str]:
        return clean_text(text).split()

    def build_vocabulary(self, corpus: Iterable[str]) -> None:
        frequencies: Counter[str] = Counter()
        for sentence in corpus:
            frequencies.update(self.tokenize(sentence))

        for token, count in frequencies.items():
            if count >= self.freq_threshold and token not in self.stoi:
                index = len(self.itos)
                self.stoi[token] = index
                self.itos[index] = token

    def numericalize(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.UNK) for token in tokens]

    def decode(self, indices: Iterable[int]) -> str:
        words: list[str] = []
        for index in indices:
            token = self.itos.get(int(index), "<UNK>")
            if token in {"<PAD>", "<SOS>", "<EOS>"}:
                continue
            words.append(token)
        return " ".join(words)


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text).lower()
    value = value.replace("findings:", "")
    value = "".join(char if char.isalnum() or char.isspace() else " " for char in value)
    return " ".join(value.split())


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(229, 236, 243, 0.52), rgba(248, 250, 252, 0.98) 45%),
                    linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
                color: #111827;
            }

            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            #MainMenu,
            footer {
                background: #ffffff !important;
            }

            header[data-testid="stHeader"] {
                box-shadow: none !important;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12) !important;
            }

            div[data-testid="stToolbar"] *,
            header[data-testid="stHeader"] * {
                color: #111827 !important;
            }

            .stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp small,
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
                color: #111827 !important;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }

            .hero {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(148, 163, 184, 0.22);
                box-shadow: 0 16px 34px rgba(15, 23, 42, 0.08);
                backdrop-filter: blur(12px);
                border-radius: 28px;
                padding: 1.6rem 1.6rem 1.4rem 1.6rem;
                margin-bottom: 1.25rem;
            }

            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.2em;
                font-size: 0.74rem;
                color: #2563eb;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }

            .hero h1 {
                margin: 0;
                font-size: 2.25rem;
                line-height: 1.05;
                color: #0f172a;
            }

            .hero p {
                margin: 0.7rem 0 0 0;
                max-width: 72ch;
                color: #1f2937;
                font-size: 1rem;
            }

            .glass-card {
                background: rgba(255, 255, 255, 0.98);
                border: 1px solid rgba(148, 163, 184, 0.20);
                border-radius: 24px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
                padding: 1rem 1rem 0.25rem 1rem;
                margin-bottom: 1rem;
            }

            .caption-box {
                background: #ffffff;
                border: 1px solid rgba(37, 99, 235, 0.18);
                border-radius: 18px;
                padding: 1rem;
                min-height: 120px;
                white-space: pre-wrap;
                word-break: break-word;
                color: #111827 !important;
            }

            .caption-box strong {
                display: block;
                margin-bottom: 0.55rem;
                color: #1d4ed8;
            }

            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div,
            div[data-testid="stSidebar"] {
                background: #ffffff !important;
                border-right: 1px solid rgba(148, 163, 184, 0.14);
            }

            section[data-testid="stSidebar"] *,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] div,
            div[data-testid="stSidebar"] *,
            div[data-testid="stSidebar"] label,
            div[data-testid="stSidebar"] p,
            div[data-testid="stSidebar"] span,
            div[data-testid="stSidebar"] div {
                color: #111827 !important;
            }

            section[data-testid="stSidebar"] input,
            section[data-testid="stSidebar"] textarea,
            section[data-testid="stSidebar"] [role="radiogroup"],
            section[data-testid="stSidebar"] [data-baseweb="select"],
            div[data-testid="stSidebar"] input,
            div[data-testid="stSidebar"] textarea,
            div[data-testid="stSidebar"] [role="radiogroup"],
            div[data-testid="stSidebar"] [data-baseweb="select"] {
                background: #ffffff !important;
                color: #111827 !important;
            }

            section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
            section[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
            section[data-testid="stSidebar"] [data-testid="stRadio"],
            div[data-testid="stSidebar"] [data-testid="stTextInput"] input,
            div[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
            div[data-testid="stSidebar"] [data-testid="stRadio"] {
                background: #ffffff !important;
                color: #111827 !important;
            }

            section[data-testid="stSidebar"] .stSlider,
            section[data-testid="stSidebar"] .stSelectbox,
            section[data-testid="stSidebar"] .stRadio,
            section[data-testid="stSidebar"] .stCheckbox,
            div[data-testid="stSidebar"] .stSlider,
            div[data-testid="stSidebar"] .stSelectbox,
            div[data-testid="stSidebar"] .stRadio,
            div[data-testid="stSidebar"] .stCheckbox {
                background: #ffffff !important;
                border-radius: 14px;
            }

            section[data-testid="stSidebar"] .stButton > button,
            div[data-testid="stSidebar"] .stButton > button {
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                color: #ffffff !important;
            }

            div[data-testid="stFileUploader"],
            div[data-testid="stFileUploader"] *,
            div[data-testid="stFileUploaderDropzone"],
            div[data-testid="stFileUploaderDropzone"] * {
                background: #ffffff !important;
                color: #111827 !important;
            }

            div[data-testid="stFileUploaderDropzone"] {
                border: 1px solid rgba(148, 163, 184, 0.22) !important;
                box-shadow: none !important;
            }

            div[data-testid="stAlert"],
            div[data-testid="stAlert"] * {
                color: #0f172a !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 0.35rem;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 14px;
                padding: 0.75rem 1rem;
            }

            .stButton > button {
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                color: white;
                border: none;
                border-radius: 999px;
                padding: 0.6rem 1rem;
                box-shadow: 0 10px 18px rgba(37, 99, 235, 0.18);
            }

            .stButton > button:hover {
                border: none;
                color: white;
                filter: brightness(1.03);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def resolve_path(
    user_input: str,
    local_default: Path,
) -> Path:
    candidates: list[Path] = []

    if user_input.strip():
        candidates.append(Path(user_input.strip()).expanduser())

    candidates.append(local_default)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Không tìm thấy đường dẫn hợp lệ. Đã thử: {', '.join(str(item) for item in candidates)}"
    )


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_resource(show_spinner=False)
def build_vocab(train_csv: str, freq_threshold: int) -> Vocabulary:
    frame = pd.read_csv(train_csv, usecols=["text"])
    if "text" not in frame.columns:
        raise KeyError("Cột 'text' không tồn tại trong tệp CSV huấn luyện")

    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(frame["text"].fillna("").astype(str).tolist())
    return vocab


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key.removeprefix("module.")] = value
        else:
            normalized[key] = value
    return normalized


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("best_model", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return cast(dict[str, torch.Tensor], checkpoint)
    raise TypeError("Không thể đọc state dict từ tệp checkpoint")


def build_model(
    spec: ModelSpec, vocab_size: int, device: torch.device
) -> torch.nn.Module:
    if spec.architecture == "lstm":
        model = LSTMImageCaptioningModel(
            vocab_size=vocab_size,
            embed_size=spec.embed_size,
            num_layers=spec.num_layers,
            dropout=spec.dropout,
            fine_tune_cnn=True,
        )
    elif spec.architecture == "transformer":
        model = TransformerImageCaptioningModel(
            vocab_size=vocab_size,
            embed_size=spec.embed_size,
            num_heads=spec.num_heads,
            num_layers=spec.num_layers,
            ff_dim=spec.ff_dim,
            dropout=spec.dropout,
            max_len=spec.max_len,
            fine_tune_cnn=True,
        )
    else:
        raise ValueError(f"Kiến trúc không hỗ trợ: {spec.architecture}")

    checkpoint = torch.load(spec.checkpoint, map_location=device)
    state_dict = normalize_state_dict(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMAGE_MEAN], std=[IMAGE_STD]),
        ]
    )


def load_pil_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def caption_from_image(
    model: torch.nn.Module,
    image: Image.Image,
    vocab: Vocabulary,
    device: torch.device,
    beam_size: int,
    max_len: int,
    use_beam: bool,
) -> str:
    tensor = image_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        if use_beam and hasattr(model, "beam_search"):
            tokens = model.beam_search(
                tensor,
                vocab,
                beam_size=beam_size,
                max_len=max_len,
            )
        else:
            tokens = model.generate_caption(
                tensor,
                vocab,
                max_len=max_len,
            )

    return " ".join(tokens).strip()


def safe_text(value: object, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    return str(value)


def render_caption_panel(
    title: str,
    image: Image.Image,
    reference: str,
    prediction: str,
    meta: dict[str, str],
) -> None:
    st.markdown(f"### {title}")
    left, right = st.columns([1.1, 1])
    with left:
        st.image(image, use_container_width=True)
    with right:
        escaped_reference = html.escape(reference or "Không có mô tả tham chiếu.")
        escaped_prediction = html.escape(prediction or "Chưa có kết quả.")
        st.markdown(
            f"""
            <div class='caption-box'>
                <strong>Mô tả tham chiếu</strong>
                {escaped_reference}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='caption-box' style='margin-top:0.85rem;'>
                <strong>Mô tả dự đoán</strong>
                {escaped_prediction}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if meta:
            meta_text = " | ".join(
                f"{key}: {value}" for key, value in meta.items() if value
            )
            if meta_text:
                st.caption(meta_text)


def main() -> None:
    st.set_page_config(
        page_title="Ứng dụng sinh mô tả ảnh",
        page_icon="🤫",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Ứng dụng sinh mô tả ảnh</div>
            <h1>Ứng dụng sinh mô tả cho ảnh X-quang</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Cấu hình")
        architecture = st.radio("Kiến trúc", tuple(MODEL_SPECS.keys()), index=0)
        spec = MODEL_SPECS[architecture]

        checkpoint_input = st.text_input("Tệp checkpoint", value=str(spec.checkpoint))
        train_csv_input = st.text_input(
            "Tệp CSV huấn luyện",
            value=str(DEFAULT_TRAIN_CSV) if DEFAULT_TRAIN_CSV.exists() else "",
        )

        use_beam = st.checkbox("Dùng beam search", value=True)
        beam_size = st.slider("Kích thước beam", min_value=1, max_value=8, value=5)
        max_len = st.slider(
            "Độ dài mô tả tối đa", min_value=20, max_value=120, value=DEFAULT_MAX_LEN
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint_path = resolve_path(
            checkpoint_input,
            spec.checkpoint,
        )
        train_csv_path = resolve_path(
            train_csv_input,
            DEFAULT_TRAIN_CSV,
        )
        vocab = build_vocab(str(train_csv_path), FREQ_THRESHOLD)
        spec = ModelSpec(
            label=spec.label,
            architecture=spec.architecture,
            checkpoint=checkpoint_path,
            embed_size=spec.embed_size,
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            ff_dim=spec.ff_dim,
            dropout=spec.dropout,
            max_len=spec.max_len,
        )
        model = build_model(spec, len(vocab), device)
    except Exception as exc:
        st.error(
            "Không thể khởi tạo model / từ vựng. Hãy kiểm tra tệp checkpoint hoặc tệp CSV huấn luyện."
        )
        st.exception(exc)
        st.stop()

    st.markdown("### Tổng quan")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Từ vựng", f"{len(vocab):,}")
    metric_cols[1].metric("Thiết bị", device.type.upper())
    metric_cols[2].metric("Kiến trúc", spec.label)
    metric_cols[3].metric("Trọng số", checkpoint_path.name)

    st.markdown("### Tải ảnh để sinh mô tả")
    st.caption("Bạn chỉ cần tải ảnh lên để sinh mô tả.")
    uploaded_file = st.file_uploader(
        "Tải ảnh lên",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )
    source_file = uploaded_file

    if source_file is not None:
        image = Image.open(source_file).convert("L")
        left, right = st.columns([1.05, 1])
        with left:
            st.image(image, use_container_width=True)
        with right:
            st.success("Ảnh đã sẵn sàng. Bấm Sinh mô tả để tạo mô tả.")
            if st.button("Sinh mô tả", type="primary", use_container_width=True):
                prediction = caption_from_image(
                    model=model,
                    image=image,
                    vocab=vocab,
                    device=device,
                    beam_size=beam_size,
                    max_len=max_len,
                    use_beam=use_beam,
                )
                st.markdown(
                    f"""
                    <div class='caption-box'>
                        <strong>Mô tả dự đoán</strong>
                        {html.escape(prediction or 'Chưa sinh được mô tả.')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.info("Hãy tải ảnh lên để bắt đầu.")


if __name__ == "__main__":
    main()
