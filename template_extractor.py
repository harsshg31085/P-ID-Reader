import cv2
import numpy as np
from pdf2image import convert_from_path
import os

class TemplateExtractor:
    def __init__(self):
        self.coordinates = []
        self.current_image = None
        self.original_image = None
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.window_name = "Template Extractor"
        self.images = []
        self.current_page = 0
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.current_image.copy()
                cv2.rectangle(img_copy, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_x, end_y = x, y
            width = abs(end_x - self.start_x)
            height = abs(end_y - self.start_y)
            
            if width > 10 and height > 10:
                x1 = min(self.start_x, end_x)
                y1 = min(self.start_y, end_y)
                self.coordinates.append((x1, y1, width, height))
                
                cv2.rectangle(self.current_image, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
                cv2.putText(self.current_image, f"{len(self.coordinates)}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(self.window_name, self.current_image)
    
    def resize_image(self, image, max_width=1200, max_height=800):
        h, w = image.shape[:2]
        if w <= max_width and h <= max_height:
            return image
        scale = min(max_width/w, max_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    
    def load_page(self, page_num):
        if page_num < 0 or page_num >= len(self.images):
            return False
        
        pil_image = self.images[page_num]
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.original_image = opencv_image.copy()
        self.current_image = self.resize_image(opencv_image)
        self.coordinates = []
        self.current_page = page_num
        
        cv2.imshow(self.window_name, self.current_image)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        print(f"Page {page_num + 1}/{len(self.images)}")
        return True
    
    def save_templates(self, output_dir, page_num):
        scale_x = self.original_image.shape[1] / self.current_image.shape[1]
        scale_y = self.original_image.shape[0] / self.current_image.shape[0]
        
        for i, (x, y, w, h) in enumerate(self.coordinates):
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            orig_w = int(w * scale_x)
            orig_h = int(h * scale_y)
            
            roi = self.original_image[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            template_name = f"symbol_p{page_num+1}_{i+1}.png"
            template_path = os.path.join(output_dir, template_name)
            cv2.imwrite(template_path, binary)
            
            print(f"Saved: {template_name}")
    
    def extract_from_pdf(self, pdf_path, output_dir):
        print("Loading PDF...")
        self.images = convert_from_path(pdf_path, dpi=150)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not self.load_page(0):
            return
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('s'):
                self.save_templates(output_dir, self.current_page)
                print(f"Saved {len(self.coordinates)} templates")
                
            elif key == ord('n'):
                if self.current_page < len(self.images) - 1:
                    if not self.load_page(self.current_page + 1):
                        break
                else:
                    print("Last page")
                    
            elif key == ord('p'):
                if self.current_page > 0:
                    if not self.load_page(self.current_page - 1):
                        break
                else:
                    print("First page")
                    
            elif key == ord('c'):
                self.coordinates = []
                self.current_image = self.resize_image(self.original_image.copy())
                cv2.imshow(self.window_name, self.current_image)
                print("Cleared")
                
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print(f"Extraction complete. Templates in: {output_dir}")

if __name__ == "__main__":
    extractor = TemplateExtractor()
    extractor.extract_from_pdf("t.pdf", "./templates")