EXPLAIN.md - Giải thích chi tiết (phiên bản tiếng Việt)
=======================================================

Mục tiêu: tạo chatbot hỏi đáp cho sinh viên HUTECH, dùng PhoBERT để hiểu tiếng Việt, sau đó phân loại intent bằng LogisticRegression.

1) intents.json
----------------
- Là dữ liệu huấn luyện chính. Mỗi mục 'intent' gồm:
  - tag: tên intent (nhãn)
  - patterns: các câu mẫu người dùng có thể nói (dùng để train)
  - responses: các câu trả lời có thể dùng khi intent này được nhận dạng
- Thêm nhiều 'patterns' đa dạng cho mỗi intent để tăng khả năng nhận dạng.

2) train_phobert.py (giải thích từng phần)
-----------------------------------------
- import ...: chúng ta dùng transformers để tải PhoBERT, sklearn để huấn luyện classifier.
- load_intents(): đọc intents.json
- get_phobert(): tải tokenizer và model 'vinai/phobert-base'
- embed_sentence(...): token hóa câu, chạy model, thực hiện mean pooling trên last_hidden_state
  (chia cho attention mask để bỏ padding).
- build_embeddings(...): tạo embedding cho toàn bộ patterns (mỗi pattern -> 1 embedding)
- LabelEncoder: chuyển các tag (chuỗi) thành số (ví dụ 'hoc_phi' -> 0)
- LogisticRegression: huấn luyện classifier trên embeddings X và nhãn y
- Lưu artifacts bằng joblib: classifier.joblib, label_encoder.joblib và một file metadata
  chứa tên model PhoBERT để load trở lại trong inference.

3) model_utils.py
------------------
- models_exist(): kiểm tra xem artifacts đã tồn tại hay chưa
- load_classifier(): load classifier + label encoder + tokenizer + phobert model
- embed_sentence(): hàm giống hệt trong train để đảm bảo consistency

4) app.py (Flask web)
---------------------
- Khi app khởi động, nó load intents.json và cố gắng load models (nếu chưa có sẽ thông báo).
- Route '/': trả về giao diện chat.
- Route '/api/chat' (POST): nhận JSON {'message': '...'} -> trả về {'reply': '...'}
  + Bên trong, get_response() sẽ embed câu hỏi, dùng classifier.predict để lấy tag,
    sau đó chọn 1 response tương ứng trong intents.json.
- Route '/health': trả về JSON trạng thái server và thông tin models_loaded.

5) Cách train & chạy (step-by-step)
-----------------------------------
1. Tạo virtualenv & active:
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   .\\venv\\Scripts\\Activate.ps1  # Windows PowerShell
2. Cài dependencies:
   pip install -r requirements.txt
3. Huấn luyện:
   python train_phobert.py
   - Lần đầu sẽ tải PhoBERT từ HuggingFace (cần internet).
   - Sau khi xong, kiểm tra thư mục models/ có classifier.joblib, label_encoder.joblib, phobert_meta.json
4. Chạy Flask:
   python app.py
   Mở http://127.0.0.1:5000 để chat.

6) Mẹo tăng chất lượng
------------------------
- Thêm nhiều patterns (ít nhất 10-20 mẫu) cho mỗi intent.
- Thu thập log các câu hỏi thực tế (luôn hỏi người dùng "có hữu ích không?"), dán nhãn và train lại.
- Nếu có GPU, cân nhắc fine-tune PhoBERT trực tiếp (sẽ tốt hơn nhiều nhưng cần GPU).

7) Các điểm cần chú ý kỹ thuật
-------------------------------
- Mean pooling: trung bình hóa các hidden states theo attention mask để loại bỏ padding.
- Nếu muốn nhanh hơn & nhẹ hơn: dùng sentence-transformers hoặc precomputed lightweight embeddings.
- Nếu gặp lỗi memory khi tải model, cân nhắc dùng model nhỏ hơn hoặc chuyển train sang máy có GPU.
