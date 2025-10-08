import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import json
import time
import re

# Cấu hình
START_URL = "https://www.hutech.edu.vn/"
DOMAIN = "hutech.edu.vn"
MAX_PAGES = 10  # crawl tối đa 500 trang

visited = set()
queue = deque([START_URL])
data = []

def is_internal(url):
    return DOMAIN in urlparse(url).netloc

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_tag(title, text):
    text = (title + " " + text).lower()
    if "học phí" in text: return "hoc_phi"
    if "đăng ký" in text or "học phần" in text: return "dang_ky_mon"
    if "thông báo" in text: return "thong_bao"
    if "lịch học" in text or "thời khóa biểu" in text: return "lich_hoc"
    if "điểm" in text or "kết quả học tập" in text: return "diem"
    if "học bổng" in text: return "hoc_bong"
    if "tuyển sinh" in text: return "tuyen_sinh"
    if "khoa" in text and "hutech" in text: return "gioi_thieu_khoa"
    return "khac"

# Bắt đầu crawl
while queue and len(visited) < MAX_PAGES:
    url = queue.popleft()
    if url in visited:
        continue

    try:
        res = requests.get(url, timeout=10)
        res.encoding = "utf-8"
        if res.status_code != 200 or "text/html" not in res.headers.get("content-type", ""):
            continue
    except Exception as e:
        print("❌ Lỗi khi truy cập:", url, "->", e)
        continue

    visited.add(url)
    soup = BeautifulSoup(res.text, "html.parser")

    title = soup.title.text.strip() if soup.title else ""
    paragraphs = [clean_text(p.text) for p in soup.find_all("p") if len(p.text.strip()) > 40]
    text_content = " ".join(paragraphs)

    if title and text_content:
        tag = detect_tag(title, text_content)
        data.append({
            "tag": tag,
            "patterns": [title],
            "responses": [text_content[:400] + "..."]
        })
        print(f"✅ {len(visited)}. {title[:70]}")

    for a in soup.find_all("a", href=True):
        link = urljoin(url, a['href'])
        if is_internal(link) and link not in visited and len(queue) < 1000:
            queue.append(link)

    time.sleep(0.5)

print(f"\n🌐 Đã crawl {len(visited)} trang, chuẩn bị lưu file...")

# Xuất intents.json
with open("intents.json", "w", encoding="utf-8") as f:
    json.dump({"intents": data}, f, ensure_ascii=False, indent=2)

print("✅ Đã lưu vào intents.json")
