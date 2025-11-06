import cv2
import numpy as np
from keras.models import load_model
import os
import random
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading

# Lấy đường dẫn thư mục hiện tại
script_dir = os.path.dirname(os.path.abspath(__file__))

# Tạo đường dẫn đầy đủ cho các file cần thiết
model_path = os.path.join(script_dir, 'model_file_30epochs.h5')
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')

# Tải model và bộ phát hiện khuôn mặt
model = load_model(model_path)
faceDetect = cv2.CascadeClassifier(cascade_path)

# Nhãn cho các cảm xúc
emotion_labels = {0:'Tuc gian', 1:'Ghe tom', 2:'So hai', 3:'Hanh phuc', 
                 4:'Binh thuong', 5:'Buon ba', 6:'Ngac nhien'}

# Hàm dự đoán tình yêu dựa trên chuỗi cảm xúc
def predict_love(emotion_sequence, duration=5):
    if not emotion_sequence:
        return "Không đủ dữ liệu để dự đoán"
    
    # Lọc nhiễu bằng cách chỉ xét các cảm xúc xuất hiện liên tiếp
    stable_emotions = []
    current_emotion = emotion_sequence[0]
    count = 1
    
    for emotion in emotion_sequence[1:]:
        if emotion == current_emotion:
            count += 1
        else:
            if count >= 3:  # Chỉ lấy cảm xúc xuất hiện liên tiếp >= 3 lần
                stable_emotions.extend([current_emotion] * count)
            current_emotion = emotion
            count = 1
    
    # Thêm cảm xúc cuối cùng nếu đủ điều kiện
    if count >= 3:
        stable_emotions.extend([current_emotion] * count)
    
    # Nếu không có cảm xúc ổn định nào, sử dụng toàn bộ chuỗi
    if not stable_emotions:
        stable_emotions = emotion_sequence
    
    # Đếm số lần xuất hiện của mỗi cảm xúc
    emotion_counts = {}
    for emotion in stable_emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    total = len(stable_emotions)
    
    # Tính tỷ lệ phần trăm các cảm xúc
    happy_percent = emotion_counts.get('Hanh phuc', 0) * 100 / total
    neutral_percent = emotion_counts.get('Binh thuong', 0) * 100 / total
    sad_percent = emotion_counts.get('Buon ba', 0) * 100 / total
    surprised_percent = emotion_counts.get('Ngac nhien', 0) * 100 / total
    angry_percent = emotion_counts.get('Tuc gian', 0) * 100 / total
    
    # Tính điểm tích cực
    positive_score = (happy_percent * 1.5 + surprised_percent * 0.8 + neutral_percent * 0.5) / (1.5 + 0.8 + 0.5)
    # Tính điểm tiêu cực
    negative_score = (sad_percent * 1.2 + angry_percent * 1.0) / (1.2 + 1.0)
    
    # Các thông điệp dự đoán tình yêu
    love_messages = {
        'very_positive': [ # Rất tích cực
            "💖 Tình yêu đang nở rộ! Hãy nắm bắt cơ hội này!",
            "💘 Cupid đã nhắm trúng tim bạn rồi!",
            "💑 Một mối quan hệ tuyệt vời đang chờ đợi!",
        ],
        'positive': [ # Tích cực
            "💝 Tình yêu đang đến gần, hãy mở lòng đón nhận!",
            "🌹 Những dấu hiệu tích cực trong chuyện tình cảm!",
            "💌 Có người đang thầm thương trộm nhớ bạn đấy!",
        ],
        'neutral': [ # Bình thường
            "💭 Hãy kiên nhẫn, tình yêu cần thời gian!",
            "🤔 Dành thời gian để hiểu rõ cảm xúc của mình!",
            "🌱 Tình yêu đang dần hình thành!",
        ],
        'negative': [ # Tiêu cực
            "💔 Có thể bạn cần thêm thời gian cho bản thân!",
            "🍂 Đừng vội vàng, hãy để mọi thứ tự nhiên!",
            "🌈 Sau cơn mưa trời lại sáng!",
        ]
    }
    
    # Chọn thông điệp dựa trên điểm số tổng hợp
    if positive_score > 70 and negative_score < 20:
        prediction = random.choice(love_messages['very_positive'])
    elif positive_score > 50 and negative_score < 30:
        prediction = random.choice(love_messages['positive'])
    elif positive_score > 30 or (positive_score > 20 and negative_score < 40):
        prediction = random.choice(love_messages['neutral'])
    else:
        prediction = random.choice(love_messages['negative'])
        
    return prediction

# Tạo class cho ứng dụng
class LovePredictor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dự Đoán Tình Yêu Qua Cảm Xúc 💝")
        self.root.geometry("1000x800")
        
        # Thiết lập style với font hỗ trợ tiếng Việt
        self.style = ttk.Style()
        self.style.configure("Custom.TFrame", background="#f0f0f0")
        self.style.configure("Custom.TLabel", background="#f0f0f0", 
                           font=("Times New Roman", 12))
        self.style.configure("Title.TLabel", background="#f0f0f0", 
                           font=("Times New Roman", 24, "bold"), justify="center")
        self.style.configure("ResultTitle.TLabel", background="#f0f0f0", 
                           font=("Times New Roman", 18, "bold"), justify="center")
        self.style.configure("Emotion.TLabel", background="#f0f0f0", 
                           font=("Times New Roman", 14, "bold"))
        self.style.configure("TButton", font=("Times New Roman", 12, "bold"))
        
        # Style mới cho labels của progress bar
        self.style.configure("Status.TLabel", 
                           background="#f0f0f0",
                           font=("Times New Roman", 12, "bold"),
                           foreground="#FF69B4")
        self.style.configure("Percent.TLabel",
                           background="#f0f0f0",
                           font=("Times New Roman", 12, "bold"),
                           foreground="#FF69B4")
        
        # Bind event để cập nhật scroll region
        self.root.bind("<Configure>", self.on_window_configure)
        
        # Bind mousewheel cho toàn bộ cửa sổ
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)
        
        # Khởi tạo biến
        self.video = None
        self.emotion_sequence = []
        self.is_running = False
        self.current_emotion = ""
        self.prediction_text = ""
        
        self.setup_gui()
        
    def setup_gui(self):
        # Tạo container chính với thanh cuộn
        container = ttk.Frame(self.root)
        container.pack(expand=True, fill="both")
        
        # Tạo canvas và scrollbar cho toàn bộ màn hình
        self.main_canvas = tk.Canvas(container, bg="#f0f0f0")
        main_scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.main_canvas.yview)
        
        # Frame chính có thể cuộn
        main_frame = ttk.Frame(self.main_canvas, style="Custom.TFrame")
        
        # Cấu hình canvas
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        # Đóng gói scrollbar và canvas
        main_scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", expand=True, fill="both")
        
        # Tạo window trong canvas và căn giữa
        self.main_canvas.create_window((500, 0), window=main_frame, anchor="n")
        
        # Container cho nội dung chính
        content_frame = ttk.Frame(main_frame, style="Custom.TFrame")
        content_frame.pack(pady=20, padx=20)
        
        # Tiêu đề
        title = ttk.Label(content_frame, text="Phân Tích Cảm Xúc & Dự Đoán Tình Yêu", 
                         style="Title.TLabel")
        title.pack(pady=(0,20))
        
        # Frame video
        self.video_frame = ttk.Frame(content_frame, style="Custom.TFrame")
        self.video_frame.pack(pady=(0,20))
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Frame thông tin
        info_frame = ttk.Frame(content_frame, style="Custom.TFrame")
        info_frame.pack(pady=(0,20), fill="x")
        
        # Progress bar frame với style hiện đại
        progress_frame = ttk.Frame(info_frame, style="Custom.TFrame")
        progress_frame.pack(pady=(0,10))
        
        # Frame cho progress bar và label
        progress_container = ttk.Frame(progress_frame, style="Custom.TFrame")
        progress_container.pack(pady=5)
        
        # Style cho progress bar
        self.style.configure("Modern.Horizontal.TProgressbar",
                           troughcolor='#f0f0f0',
                           background='#FF69B4',
                           thickness=12,
                           borderwidth=0)
        
        # Status Label phía trên progress bar
        self.status_label = ttk.Label(progress_container, text="", style="Status.TLabel")
        self.status_label.pack(pady=(0, 5))
        
        # Progress bar hiện đại
        self.progress_bar = ttk.Progressbar(progress_container,
                                          style="Modern.Horizontal.TProgressbar",
                                          length=400,
                                          mode='determinate')
        self.progress_bar.pack(pady=(0,5))
        
        # Frame cho label hiển thị phần trăm
        label_frame = ttk.Frame(progress_frame, style="Custom.TFrame")
        label_frame.pack(fill='x')
        
        # Label phần trăm
        self.percent_label = ttk.Label(label_frame, text="0%", style="Percent.TLabel")
        self.percent_label.pack(pady=(5,0))
        
        # Label cho phần trăm
        self.percent_label = ttk.Label(label_frame, text="0%",
                                     style="Percent.TLabel")
        self.percent_label.pack(side='right', padx=(0,10))
        
        # Label cho trạng thái
        self.progress_label = ttk.Label(label_frame, text="Đang chuẩn bị...",
                                      style="Status.TLabel")
        self.progress_label.pack(side='left', padx=(10,0))
        
        # Cảm xúc hiện tại
        self.emotion_label = ttk.Label(info_frame, text="Cảm xúc: ", 
                                     style="Emotion.TLabel")
        self.emotion_label.pack(pady=(0,10))
        
        # Frame kết quả dự đoán
        result_frame = ttk.Frame(content_frame, style="Custom.TFrame")
        result_frame.pack(pady=(0,20), fill="x")
        
        # Tiêu đề kết quả
        self.result_title = ttk.Label(result_frame, 
                                    text="",  # Ban đầu để trống
                                    style="ResultTitle.TLabel")
        self.result_title.pack(pady=(0,10))
        
        # Dự đoán
        self.prediction_label = ttk.Label(result_frame, text="", 
                                        style="Custom.TLabel", wraplength=580,
                                        justify="center")
        self.prediction_label.pack(pady=5, padx=10)
        
        # Tùy chỉnh style cho prediction label
        self.style.configure("Prediction.TLabel", 
                           background="#f0f0f0", 
                           font=("Times New Roman", 16),
                           justify="center",
                           wraplength=580)
        self.prediction_label.configure(style="Prediction.TLabel")
        
        # Frame điều khiển
        control_frame = ttk.Frame(content_frame, style="Custom.TFrame")
        control_frame.pack(pady=(0,20))
        
        # Nút Bắt đầu/Dừng
        self.start_button = ttk.Button(control_frame, text="Bắt đầu", 
                                     command=self.toggle_camera,
                                     width=20)  # Đặt độ rộng cố định cho nút
        self.start_button.pack()
        
    def toggle_camera(self):
        if not self.is_running:
            # Bắt đầu phiên mới
            self.start_camera()
            self.start_button.config(text="Dừng")
        else:
            # Nếu đang chạy và ấn dừng
            if not self.prediction_shown:
                # Nếu chưa hoàn thành phân tích
                self.stop_camera()
                self.start_button.config(text="Bắt đầu")
            else:
                # Nếu đã hoàn thành phân tích và ấn "Bắt đầu lại"
                self.stop_camera()
                self.start_camera()
                self.start_button.config(text="Dừng")
            
    def start_camera(self):
        self.video = cv2.VideoCapture(0)
        self.is_running = True
        self.emotion_sequence = []
        self.start_time = datetime.now()
        self.prediction_shown = False
        self.last_face = None  # Thêm biến để lưu thông tin khuôn mặt cuối cùng
        self.last_frame = None  # Thêm biến để lưu frame cuối cùng
        
        # Reset tất cả các hiển thị
        self.progress_bar['value'] = 0
        self.percent_label.config(text="0%")
        self.status_label.config(text="Đang chuẩn bị phân tích...")
        self.emotion_label.config(text="Cảm xúc: ")
        self.result_title.config(text="")
        self.prediction_label.config(text="")
        
        self.update_frame()
        
    def stop_camera(self):
        self.is_running = False
        if self.video is not None:
            self.video.release()
            # Chỉ reset progress bar nếu chưa hoàn thành phân tích
            if not self.prediction_shown:
                self.progress_bar['value'] = 0
                self.percent_label.config(text="0%")
                self.status_label.config(text="Đã dừng phân tích")
            else:
                # Giữ nguyên progress bar ở 100% khi hoàn thành
                self.progress_bar['value'] = 100
                self.percent_label.config(text="100%")
                self.status_label.config(text="Đã hoàn thành phân tích!")
            
    def update_frame(self):
        if self.is_running:
            ret, frame = self.video.read()
            if ret:
                # Lưu frame hiện tại
                self.last_frame = frame.copy()
                
                # Xử lý frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceDetect.detectMultiScale(gray, 1.3, 3)
                
                # Xử lý từng khuôn mặt phát hiện được
                for x, y, w, h in faces:
                    sub_face_img = gray[y:y+h, x:x+w]
                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized/255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))
                    result = model.predict(reshaped)
                    label = emotion_labels[np.argmax(result, axis=1)[0]]
                    
                    # Lưu cảm xúc vào chuỗi theo dõi
                    self.emotion_sequence.append(label)
                    self.current_emotion = label
                    
                    # Lưu thông tin khuôn mặt cuối cùng
                    self.last_face = {
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'label': label
                    }
                
                # Tính thời gian và cập nhật progress bar
                time_elapsed = (datetime.now() - self.start_time).seconds
                remaining_time = max(5 - time_elapsed, 0)
                progress = min((time_elapsed / 5) * 100, 100)
                
                # Cập nhật progress bar và nhãn phần trăm
                self.progress_bar['value'] = progress
                self.percent_label.config(text=f"{int(progress)}%")
                
                # Cập nhật status tùy theo tiến độ
                if progress < 100:
                    self.status_label.config(text="Đang phân tích cảm xúc...")
                else:
                    self.status_label.config(text="Đã hoàn thành phân tích!")
                
                # Cập nhật progress bar
                progress = min((time_elapsed / 5.0) * 100, 100)
                self.progress_bar['value'] = progress
                if remaining_time > 0:
                    self.progress_label.config(
                        text=f"Đang phân tích: {remaining_time}s còn lại..."
                    )
                
                # Chuyển đổi frame để hiển thị trong tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Cập nhật nhãn cảm xúc
                self.emotion_label.config(text=f"Cảm xúc hiện tại:  {self.current_emotion}")
                
                # Sau 5 giây, hiện kết quả và vẽ khung detect cuối cùng
                if time_elapsed >= 5:
                    if not hasattr(self, 'prediction_shown') or not self.prediction_shown:
                        # Tính toán và hiển thị kết quả dự đoán
                        self.prediction_text = predict_love(self.emotion_sequence)
                        
                        # Hiển thị tiêu đề và kết quả dự đoán
                        self.result_title.config(text="✨ Kết quả dự đoán tình yêu ✨")
                        self.prediction_label.config(text=self.prediction_text)
                        
                        self.prediction_shown = True
                        self.progress_label.config(text="Phân tích hoàn tất! ✨")
                        
                        # Vẽ khung và nhãn cho khuôn mặt cuối cùng trên frame cuối
                        if self.last_face and self.last_frame is not None:
                            final_frame = self.last_frame.copy()
                            x = self.last_face['x']
                            y = self.last_face['y']
                            w = self.last_face['w']
                            h = self.last_face['h']
                            label = self.last_face['label']
                            
                            # Vẽ khung và nhãn trên frame cuối
                            cv2.rectangle(final_frame, (x,y), (x+w, y+h), (255,155,255), 2)
                            cv2.rectangle(final_frame, (x,y-40), (x+w, y), (255,155,255), -1)
                            cv2.putText(final_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                            
                            # Hiển thị frame cuối với khung và nhãn
                            frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            imgtk = ImageTk.PhotoImage(image=img)
                            self.video_label.imgtk = imgtk
                            self.video_label.configure(image=imgtk)
                        
                        self.stop_camera()
                        self.start_button.config(text="Bắt đầu lại")
                
                # Lặp lại hàm cập nhật
                self.root.after(10, self.update_frame)

    def on_window_configure(self, event=None):
        """Cập nhật vùng cuộn khi kích thước cửa sổ thay đổi"""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        
    def on_mousewheel(self, event):
        """Xử lý sự kiện cuộn chuột"""
        self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def update_scroll_region(self):
        """Cập nhật vùng cuộn sau khi nội dung thay đổi"""
        self.main_canvas.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    # Khởi tạo và chạy ứng dụng
if __name__ == "__main__":
    app = LovePredictor()
    app.root.mainloop()