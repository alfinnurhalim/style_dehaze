import os
import torch
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

# New Logger Class
class TrainingLogger:
    def __init__(self, log_path, log_filename='training.log', graph_filename='train_loss.png'):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, log_filename)
        self.graph_path = os.path.join(log_path, graph_filename)
        self.loss_history = []
        
        # Create a logger instance
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler and set formatter
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Avoid duplicate handlers if already set
        if not self.logger.handlers:
            self.logger.addHandler(fh)
    
    def log_epoch_loss(self, epoch, loss):
        # Convert loss (Tensor or float) to a CPU float
        if isinstance(loss, torch.Tensor):
            cpu_loss = loss.cpu().item()
        else:
            cpu_loss = float(loss)

        self.logger.info(f"Epoch {epoch} - Last Batch Loss: {cpu_loss:.4f}")
        self.loss_history.append(cpu_loss)
    
    def update_graph(self):
        plt.figure()
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(self.graph_path)
        plt.close()
        print(f"Updated training graph saved at: {self.graph_path}")

def overlay_attention_on_image(attn_map, content_img, alpha=0.5):
    """
    Overlays the attention heatmap on the content image.
    """
    attn_map = attn_map.squeeze().cpu().numpy()   # [H, W]
    attn_map = cv2.resize(attn_map, (content_img.size(2), content_img.size(1)))
    attn_norm = (attn_map * 255).astype(np.uint8)
    attn_color = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)  # [H, W, 3]

    # Convert content to [H, W, 3] numpy
    content_np = (content_img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    overlayed = cv2.addWeighted(content_np, 1 - alpha, attn_color, alpha, 0)
    return overlayed
