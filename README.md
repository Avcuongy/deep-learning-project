# Project setup

## Enviroment

- **Python Version:** Python >= 3.9

## Project setup

Run the following commands in your terminal:

```bash
git clone https://github.com/Avcuongy/deep-learning-project.git

cd deep-learning-project

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

pip install -e .

python scripts/config.py
```

## Streamlit app

Chạy ứng dụng bằng lệnh:

```bash
streamlit run streamlit_app.py
```

Ứng dụng có 2 phần:

1. Inference trên test set.
2. Dán hoặc tải ảnh lên để sinh caption.

Nếu bạn chưa có file dữ liệu local, app sẽ thử tải dữ liệu split và ảnh từ KaggleHub khi tuỳ chọn tự động tải đang bật.
