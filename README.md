Chatbot HUTECH - PhoBERT (Version 2)
===================================
Mô tả:
  - Chatbot trả lời câu hỏi sinh viên HUTECH về học phí, lịch học, đăng ký môn, điểm, thư viện...
  - Sử dụng PhoBERT (vinai/phobert-base) để sinh embedding cho câu tiếng Việt.
  - Dùng LogisticRegression (sklearn) làm classifier trên embedding vectors.
  - Flask app phục vụ giao diện web và API /api/chat.

Nội dung thư mục:
  - intents.json          : Dữ liệu intents (patterns + responses)
  - train_phobert.py      : Script huấn luyện (tạo models/classifier.joblib & models/label_encoder.joblib)
  - model_utils.py        : Hàm load model và embed câu cho inference
  - app.py                : Flask app (API + UI)
  - templates/index.html  : Giao diện chat
  - static/style.css      : CSS
  - EXPLAIN.md            : Giải thích chi tiết từng file, cách train, cách run, từng dòng mã chủ chốt
  - requirements.txt      : Danh sách packages cần cài

Hướng dẫn nhanh:
  1) Tạo virtualenv:
     python -m venv venv
     # macOS / Linux:
     source venv/bin/activate
     # Windows (PowerShell):
     .\venv\Scripts\Activate.ps1
  2) Cài packages:
     pip install -r requirements.txt
  3) Huấn luyện (lần đầu sẽ tải PhoBERT từ internet):
     python train_phobert.py
  4) Chạy Flask:
     python app.py
     Mở trình duyệt: http://127.0.0.1:5000

Lưu ý & troubleshooting:
  - Nếu bạn gặp lỗi "out of memory" khi tải mô hình trên máy yếu, hãy chạy train trên máy có GPU hoặc dùng CPU nhưng sẽ chậm.
  - Nếu muốn bỏ qua train và dùng mô hình đơn giản (TF-IDF), mình có thể thêm script alternative_train_tfidf.py.
  - Đảm bảo pip, Python đã cài và active virtualenv trước khi pip install.
