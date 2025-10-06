import tkinter as tk
from PIL import Image, ImageDraw

from pathlib import Path
import sys
(sys.path.append(str(Path(__file__).resolve().parent.parent)))
from load_model import InferenceModel

CANVAS_W, CANVAS_H = 280, 280

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Predict Handwritten Digit")

        # Tải mô hình dự đoán
        self.predictor = InferenceModel()

        # Tiêu đề
        self.title = tk.Label(root, text="Draw a digit and click 'Predict'", font=("Helvetica", 14))
        self.title.grid(row=0, column=0, columnspan=3)

        # Vùng vẽ
        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg='white')
        self.canvas.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        # Ô hiển thị kết quả
        self.result_entry = tk.Entry(root, width=10, font=("Helvetica", 24), justify='center')  
        self.result_entry.grid(row=2, column=0, columnspan=3, pady=10)
        self.result_entry.insert(0, "—")
        self.result_entry.config(state="disabled")

        # Nút lệnh
        self.clear_button = tk.Button(root, text="Clear", width=10, command=self.clear_canvas)
        self.clear_button.grid(row=3, column=0, pady=10)

        self.predict_button = tk.Button(root, text="Predict", width=10, command=self.predict_digit)
        self.predict_button.grid(row=3, column=1, pady=10)

        self.exit_button = tk.Button(root, text="Exit", width=10, command=root.destroy)
        self.exit_button.grid(row=3, column=2, pady=10)

        # Thiết lập vẽ
        self.last_x, self.last_y = None, None
        self.brush_size = 8
        self.brush_fill = 'black'

        self.canvas.bind('<ButtonPress-1>', self.pen_down)
        self.canvas.bind('<B1-Motion>', self.pen_move)
        self.canvas.bind('<ButtonRelease-1>', self.pen_up)

        # Tạo ảnh trống ban đầu
        self.clear_canvas()

    def pen_down(self, event):
        self.last_x, self.last_y = event.x, event.y

    def pen_move(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.brush_size, fill=self.brush_fill,
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=self.brush_fill, width=self.brush_size)
        self.last_x, self.last_y = x, y

    def pen_up(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """Xoá toàn bộ canvas và tạo lại ảnh nền trắng"""
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_W, CANVAS_H), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        """Xử lý ảnh và hiển thị kết quả dự đoán"""
        label = self.predictor.get_labels(self.image)
        self.result_entry.config(state="normal")
        self.result_entry.delete(0, tk.END)
        self.result_entry.insert(0, str(label.item()))
        self.result_entry.config(state="disabled")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    app.run()
