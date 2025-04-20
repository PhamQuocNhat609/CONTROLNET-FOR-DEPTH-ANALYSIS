import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageTk, ImageFilter
import cv2
from controlnet_aux import CannyDetector, MidasDetector, NormalBaeDetector
from controlnet_aux.mlsd import MLSDdetector
from controlnet_aux.open_pose import OpenposeDetector
from controlnet_aux.lineart import LineartDetector

#Khởi tạo các mô hình ControlNet
mlsd_model = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
openpose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
lineart_model = LineartDetector.from_pretrained("lllyasviel/Annotators")
midas_model = MidasDetector.from_pretrained("lllyasviel/Annotators")
normalbae_model = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

#Thiết lập màu sắc cho các nhóm nút
BUTTON_COLORS = {
    "blue": {"bg": "#3498DB", "fg": "white", "activebackground": "#2980B9"},
    "red": {"bg": "#E74C3C", "fg": "white", "activebackground": "#C0392B"},
    "green": {"bg": "#2ECC71", "fg": "white", "activebackground": "#27AE60"},
    "orange": {"bg": "#F39C12", "fg": "black", "activebackground": "#D68910"}
}
def create_button(parent, text, command, color_type="blue"):
    return tk.Button(
        parent, text=text, command=command, **BUTTON_COLORS[color_type]
    )

class ImageProcessorApp:
    def __init__(self, root):
        #Khởi tạo giao diện ứng dụng
        self.root = root
        self.root.title("Ứng Dụng Xử Lý Ảnh")
        self.root.geometry("1600x900")
        self.root.configure(bg="#2C3E50")
        
        self.image = None
        self.original_image = None

        
        #Tạo giao diện điều khiển
        control_frame = tk.Frame(root, padx=10, pady=10, bg="#34495E")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        #Khung hiển thị ảnh
        display_frame = tk.Frame(root, padx=10, pady=10, bg="#2C3E50")
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(display_frame, bg="#ECF0F1", highlightthickness=2, highlightbackground="#1ABC9C")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        #Khung tải ảnh
        file_frame = tk.LabelFrame(control_frame, text="Ảnh", padx=5, pady=5, bg="#1ABC9C", fg="white", font=("Arial", 10, "bold"))
        file_frame.pack(fill="x", pady=5)

        create_button(file_frame, "📂 Tải Ảnh", self.load_image, "green").pack(fill="x", pady=2)
        create_button(file_frame, "📷 Chụp Ảnh", self.capture_image, "orange").pack(fill="x", pady=2)
        
        #Khung ControlNet
        controlnet_frame = tk.LabelFrame(control_frame, text="ControlNet", padx=5, pady=5, bg="#1ABC9C", fg="white", font=("Arial", 10, "bold"))
        controlnet_frame.pack(fill="x", pady=5)
        
        self.detectors = {
            "Canny": CannyDetector(),
            "MLSD": mlsd_model,
            "OpenPose": openpose_model,
            "Lineart": lineart_model,
            "Midas": midas_model,
            "NormalBae": normalbae_model
        }
        
        for name in self.detectors:
            create_button(controlnet_frame, name, lambda n=name: self.apply_controlnet(n), "blue").pack(fill="x", pady=2)

        #Khung chỉnh sửa ảnh
        adjust_frame = tk.LabelFrame(control_frame, text="Chỉnh Sửa Ảnh", padx=5, pady=5, bg="#1ABC9C", fg="white", font=("Arial", 10, "bold"))
        adjust_frame.pack(fill="x", pady=5)
        
        self.create_slider(adjust_frame, "Độ Sắc Nét", 1.0, 5.0, 1.0, "sharpness")
        self.create_slider(adjust_frame, "Độ Sáng", 0.5, 2.0, 1.0, "brightness")
        self.create_slider(adjust_frame, "Độ Tương Phản", 0.5, 2.0, 1.0, "contrast")
        self.create_slider(adjust_frame, "Độ Bão Hòa", 0.5, 2.0, 1.0, "saturation")
        self.create_slider(adjust_frame, "Độ Mờ", 0, 10, 0, "blur")
        
        action_frame = tk.Frame(control_frame, pady=5, bg="#34495E")
        action_frame.pack(fill="x")
        
        create_button(action_frame, "✔ Áp Dụng", self.apply_changes, "green").pack(fill="x", pady=2)
        create_button(action_frame, "💾 Lưu Ảnh", self.save_image, "orange").pack(fill="x", pady=2)
        create_button(action_frame, "🔄 Khôi Phục", self.reset_image, "blue").pack(fill="x", pady=2)
        create_button(action_frame, "🗑 Xóa Ảnh", self.delete_image, "red").pack(fill="x", pady=2)

    #Thanh trượt
    def create_slider(self, parent, label, min_val, max_val, blue, var_name):
        frame = tk.Frame(parent, bg="#1ABC9C")
        frame.pack(fill="x")
        tk.Label(frame, text=label, bg="#1ABC9C", fg="white").pack(anchor="w")
        scale = tk.Scale(frame, from_=min_val, to=max_val, resolution=0.1, orient=tk.HORIZONTAL, bg="#ECF0F1")
        scale.set(blue)
        scale.pack(fill="x")
        setattr(self, var_name + "_scale", scale)
    
    #Load ảnh
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return

        self.original_image = Image.open(file_path)

        # Lấy kích thước canvas hiện tại
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Resize ảnh sao cho vừa với khung hình
        self.image = self.original_image.copy()
        self.image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        # Chuyển đổi ảnh sang định dạng Tkinter
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Xóa nội dung cũ và đặt ảnh mới vào giữa Canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_image, anchor=tk.CENTER)
        self.canvas.image = self.tk_image  # Giữ tham chiếu để tránh ảnh bị xóa

    #Chụp ảnh
    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở webcam.")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(frame)
            self.image = self.original_image.copy()
            self.show_image()
            
    #Chỉnh sửa ảnh
    def apply_changes(self):
        if self.image is None:
            messagebox.showerror("Lỗi", "Vui lòng tải ảnh trước!")
            return
        enhancer = ImageEnhance.Sharpness(self.image)
        self.image = enhancer.enhance(self.sharpness_scale.get())
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(self.brightness_scale.get())
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(self.contrast_scale.get())
        enhancer = ImageEnhance.Color(self.image)
        self.image = enhancer.enhance(self.saturation_scale.get())
        if self.blur_scale.get() > 0:
            self.image = self.image.filter(ImageFilter.GaussianBlur(self.blur_scale.get()))
        self.show_image()

    #Áp dụng controlnet
    def apply_controlnet(self, model_name):
        if self.image is None:
            return

        model = self.detectors.get(model_name)  # Lấy model từ dictionary
        if model is None:
            messagebox.showerror("Lỗi", f"Không tìm thấy model: {model_name}")
            return

        # Giữ nguyên kích thước ảnh gốc
        input_image = self.image.copy()

        # Nếu model yêu cầu kích thước cố định, lưu lại kích thước gốc
        original_size = input_image.size
        required_size = (512, 512)  # Thay đổi theo yêu cầu model
        input_image = input_image.resize(required_size, Image.Resampling.LANCZOS)

        # Gửi ảnh vào ControlNet
        processed_image = model(input_image)  # Kiểm tra nếu model có phương thức phù hợp

        # Resize ảnh đã xử lý về kích thước gốc
        processed_image = processed_image.resize(original_size, Image.Resampling.LANCZOS)

         # Cập nhật ảnh vào self.image để có thể lưu được
        self.image = processed_image

        # Hiển thị ảnh lên canvas
        self.show_processed_image(processed_image)

    #Lưu ảnh
    def save_image(self):
        if self.image is None:
            messagebox.showerror("Lỗi", "Vui lòng tải ảnh trước khi lưu!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"),("JPEG files", "*.jpg;*.jpeg"),("BMP files", "*.bmp"),("All Files", "*.*")])
        if file_path:
            try:
                self.image.save(file_path)
                messagebox.showinfo("Thành công", "Ảnh đã được lưu thành công!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {str(e)}")

    #Khôi phục ảnh
    def reset_image(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.show_image()
    
    #Hiện ảnh
    def show_image(self):
        if self.image is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img = self.image.copy()
        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_image, anchor=tk.CENTER)
        self.canvas.image = self.tk_image

    #Hiện ảnh sau khi được áp dụng controlnet vào ảnh
    def show_processed_image(self, processed_image):
        if processed_image is None:
            return

        # Resize ảnh về kích thước gốc
        processed_image = processed_image.resize(self.original_image.size, Image.Resampling.LANCZOS)

        # Chuyển ảnh sang định dạng Tkinter
        self.tk_image = ImageTk.PhotoImage(processed_image)

        # Hiển thị ảnh lên canvas
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.tk_image, anchor=tk.CENTER)
        self.canvas.image = self.tk_image 

    #Xóa ảnh khỏi Canvas
    def delete_image(self):
        self.canvas.delete("all")  # Xóa toàn bộ nội dung trên canvas
        self.tk_image = None  # Xóa tham chiếu đến ảnh để tránh lỗi
root = tk.Tk()
app = ImageProcessorApp(root)
root.mainloop()
