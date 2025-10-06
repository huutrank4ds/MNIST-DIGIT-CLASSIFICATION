from model.model import MyModel
import torch
from pathlib import Path
from torchvision import transforms
from PIL import ImageOps

class InferenceModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
            
    def load_model(self):
        # Xác định đường dẫn tuyệt đối tới model file
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        model_path = project_root / "model" / "best_weight" / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
        
        model = MyModel().to(self.device)
        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def get_labels(self, img):
        # Resize về 28x28 (như MNIST)
        img = img.resize((28, 28))

        # Đảo ngược màu (vì model huấn luyện chữ trắng nền đen)
        img = ImageOps.invert(img)

        # Chuyển sang tensor và chuẩn hóa về [0, 1]
        transform = transforms.Compose([
            transforms.Grayscale(),           # Đảm bảo 1 kênh
            transforms.ToTensor(),           # Chuyển sang tensor
        ])
        img = transform(img).to(self.device)  
        with torch.no_grad():
            output = self.model(img)

        labels = output.argmax(dim=1).cpu()
        return labels