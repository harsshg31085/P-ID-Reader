import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os
import argparse
import json
import pytesseract
from PIL import Image as PILImage
import fitz,io

class PIDDetector:
    def __init__(self):
        self.templates = {}
        self.thresholds = {}
        self.label_prefixes = {}
        self.pdf_doc = None
    
    def load_label_prefixes(self, prefixes_file):
        try:
            with open(prefixes_file, 'r') as f:
                self.label_prefixes = json.load(f)
            print(f"Loaded label prefixes for {len(self.label_prefixes)} symbol types")
        except FileNotFoundError:
            print(f"Warning: Prefixes file not found: {prefixes_file}")
            print("All labels will be accepted")
            self.label_prefixes = {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in prefixes file: {e}")
            self.label_prefixes = {}
        except Exception as e:
            print(f"Error loading prefixes file: {e}")
            self.label_prefixes = {}
    
    def load_thresholds(self, thresholds_file, default_threshold=0.7):
        try:
            with open(thresholds_file, 'r') as f:
                self.thresholds = json.load(f)
            
            if not isinstance(self.thresholds, dict):
                raise ValueError("Thresholds file must contain a JSON object")
            
            print(f"Loaded individual thresholds for {len([k for k in self.thresholds.keys() if k != 'default'])} symbols")
            
            if 'default' in self.thresholds:
                print(f"Default threshold: {self.thresholds['default']}")
            else:
                self.thresholds['default'] = default_threshold
                print(f"No default threshold found, using: {default_threshold}")
                
        except FileNotFoundError:
            print(f"Error: Thresholds file not found: {thresholds_file}")
            print("Please create a thresholds.json file with symbol-specific thresholds")
            print("Format: {\"symbol_name\": 0.65, \"default\": 0.7}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in thresholds file: {e}")
            raise
        except Exception as e:
            print(f"Error loading thresholds file: {e}")
            raise
    
    def load_templates(self, template_dir):
        template_files = [f for f in os.listdir(template_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for file in template_files:
            name = os.path.splitext(file)[0]
            path = os.path.join(template_dir, file)
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.templates[name] = template
                print(f"Loaded: {name} - {template.shape}")
        print(f"Total templates: {len(self.templates)}")
    
    def get_symbol_threshold(self, symbol_name):
        return self.thresholds.get(symbol_name, self.thresholds['default'])
    
    def extract_text_from_image(self, page_num):
        if self.pdf_doc is None:
            return []
        
        page = self.pdf_doc[page_num]
        results = []
        
        words = page.get_text("words")
        
        seen_positions = set()
        position_threshold = 2
        
        for word in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = word
            text = text.strip()
            
            if not text or len(text) < 1:
                continue
            
            position_key = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
            is_duplicate = False
            
            for seen_pos in seen_positions:
                if (abs(seen_pos[0] - position_key[0]) < position_threshold and
                    abs(seen_pos[1] - position_key[1]) < position_threshold and
                    abs(seen_pos[2] - position_key[2]) < position_threshold and
                    abs(seen_pos[3] - position_key[3]) < position_threshold):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            seen_positions.add(position_key)
            
            w = x1 - x0
            h = y1 - y0
            
            results.append({
                'bbox': [float(x0), float(y0), float(w), float(h)],
                'text': str(text),
                'center': [float(x0 + w/2), float(y0 + h/2)],
                'is_pdf_coords': True
            })
        
        with open('text_blocks.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"  Extracted {len(results)} text labels")
        
        if results:
            print(f"  Sample texts (first 10):")
            for i, r in enumerate(results[:10]):
                print(f"    {i+1}. '{r['text']}' at ({r['bbox'][0]:.1f}, {r['bbox'][1]:.1f})")
        
        return results
    
    def convert_pdf_to_image_coords(self, pdf_coords, pdf_page, image_size):
        pdf_width = pdf_page.rect.width
        pdf_height = pdf_page.rect.height
        img_width, img_height = image_size
        
        scale_x = img_width / pdf_width
        scale_y = img_height / pdf_height
        
        x, y, w, h = pdf_coords
        
        img_x = x * scale_x
        img_y = y * scale_y
        img_w = w * scale_x
        img_h = h * scale_y
        
        return (img_x, img_y, img_w, img_h)
    
    def convert_image_to_pdf_coords(self, img_coords, pdf_page, image_size):
        pdf_width = pdf_page.rect.width
        pdf_height = pdf_page.rect.height
        img_width, img_height = image_size
        
        scale_x = pdf_width / img_width
        scale_y = pdf_height / img_height
        
        x, y, w, h = img_coords
        
        pdf_x = x * scale_x
        pdf_y = y * scale_y
        pdf_w = w * scale_x
        pdf_h = h * scale_y
        
        return (pdf_x, pdf_y, pdf_w, pdf_h)
    
    def find_best_matching_text(self, symbol_bbox, text_elements, symbol_name, pdf_page, image_size, max_radius=150):
        img_width, img_height = image_size
        symbol_x, symbol_y, symbol_w, symbol_h = symbol_bbox
        
        pdf_bbox = self.convert_image_to_pdf_coords(symbol_bbox, pdf_page, image_size)
        pdf_x, pdf_y, pdf_w, pdf_h = pdf_bbox
        pdf_center = (pdf_x + pdf_w/2, pdf_y + pdf_h/2)
        
        allowed_prefixes = self.label_prefixes.get(symbol_name, [])
        
        interior_texts = []
        nearby_texts = []
        
        symbol_pdf_x, symbol_pdf_y = pdf_center
        
        for text in text_elements:
            if not text.get('is_pdf_coords', False):
                continue
                
            text_pdf_bbox = text['bbox']
            tx, ty, tw, th = text_pdf_bbox
            text_pdf_center = text['center']
            
            distance = np.sqrt((symbol_pdf_x - text_pdf_center[0])**2 + (symbol_pdf_y - text_pdf_center[1])**2)
            
            text_inside_symbol = (
                tx >= pdf_x and tx + tw <= pdf_x + pdf_w and
                ty >= pdf_y and ty + th <= pdf_y + pdf_h
            )
            
            pdf_radius = max_radius * (pdf_page.rect.width / img_width)
            text_near_symbol = distance <= pdf_radius
            
            if text_inside_symbol:
                interior_texts.append((text, distance, 0))
            elif text_near_symbol:
                nearby_texts.append((text, distance, 1))
        
        if interior_texts:
            print(f"    Found {len(interior_texts)} text(s) inside symbol")
        if nearby_texts:
            print(f"    Found {len(nearby_texts)} text(s) near symbol")
        
        search_order = interior_texts + nearby_texts
        
        if not allowed_prefixes:
            if search_order:
                search_order.sort(key=lambda x: (x[2], x[1]))
                return search_order[0][0]['text']
            return None
        
        for text_item, distance, category in search_order:
            text_obj = text_item
            text_content = text_obj['text'].strip()
            
            for prefix in allowed_prefixes:
                if text_content.startswith(prefix):
                    print(f"    Matched text '{text_content}' with prefix '{prefix}'")
                    return text_content
        
        if search_order:
            search_order.sort(key=lambda x: (x[2], x[1]))
            closest_text = search_order[0][0]['text'].strip()
            print(f"    No prefix match, using closest text: '{closest_text}'")
            return closest_text
        
        return None
    
    def multi_scale_template_match(self):
        all_matches = []
        
        step = 0.05
        scales = [0.6 + i*step for i in range(17)]
        
        for name, template in self.templates.items():
            symbol_threshold = self.get_symbol_threshold(name)
            
            for scale in scales:
                try:
                    new_w = int(template.shape[1] * scale)
                    new_h = int(template.shape[0] * scale)
                    
                    if new_w < 20 or new_h < 20 or new_w > self.gray.shape[1] or new_h > self.gray.shape[0]:
                        continue
                    
                    resized_template = cv2.resize(template, (new_w, new_h))
                    result = cv2.matchTemplate(self.gray, resized_template, cv2.TM_CCOEFF_NORMED)
                    
                    locations = np.where(result >= symbol_threshold)
                    
                    for pt in zip(*locations[::-1]):
                        confidence = result[pt[1], pt[0]]
                        all_matches.append({
                            'bbox': (pt[0], pt[1], new_w, new_h),
                            'confidence': confidence,
                            'name': name,
                            'scale': scale,
                            'threshold_used': symbol_threshold
                        })
                except:
                    continue
        
        return all_matches
    
    def remove_overlapping_detections(self, matches, overlap_threshold=0.3):
        if not matches:
            return []
        
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        filtered_matches = []
        
        for match in matches:
            x1, y1, w1, h1 = match['bbox']
            overlap = False
            
            for kept_match in filtered_matches:
                x2, y2, w2, h2 = kept_match['bbox']
                dx = min(x1 + w1, x2 + w2) - max(x1, x2)
                dy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if dx > 0 and dy > 0:
                    area_intersection = dx * dy
                    area1 = w1 * h1
                    if area_intersection / area1 > overlap_threshold:
                        overlap = True
                        break
            
            if not overlap:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def save_symbols_to_json(self, matches_data, output_path):
        symbols_data = []
        
        for page_data in matches_data:
            page_num = page_data['page']
            page_matches = page_data['matches']
            
            for match in page_matches:
                symbol_info = {
                    'page': int(page_num),
                    'symbol_type': str(match['name']),
                    'label': str(match['label']),
                    'confidence': float(match['confidence']),
                    'location': {
                        'x': int(match['bbox'][0]),
                        'y': int(match['bbox'][1]),
                        'width': int(match['bbox'][2]),
                        'height': int(match['bbox'][3])
                    },
                    'scale': float(match['scale']),
                    'threshold_used': float(match['threshold_used'])
                }
                symbols_data.append(symbol_info)
        
        with open(output_path, 'w') as f:
            json.dump(symbols_data, f, indent=2)
        
        print(f"Symbols data saved to: {output_path}")

    def create_highlighted_pdf(self, pdf_path, output_path, template_dir, thresholds_file, prefixes_file):
        self.load_templates(template_dir)
        self.load_thresholds(thresholds_file)
        self.load_label_prefixes(prefixes_file)
        
        self.pdf_doc = fitz.open(pdf_path)
        
        images = convert_from_path(pdf_path, dpi=200)
        highlighted_images = []
        
        print("\nSymbol thresholds:")
        for name in self.templates.keys():
            threshold = self.get_symbol_threshold(name)
            prefixes = self.label_prefixes.get(name, ["Any"])
            print(f"  {name}: {threshold:.2f} (prefixes: {prefixes})")
        print()
        
        failed_prefix_log = []
        all_matches_data = []  
        
        for page_num, pil_image in enumerate(images):
            print(f"Processing page {page_num + 1}")
            
            pdf_page = self.pdf_doc[page_num]
            pdf_width = pdf_page.rect.width
            pdf_height = pdf_page.rect.height
            img_width, img_height = pil_image.size
            
            scale_x = img_width / pdf_width
            scale_y = img_height / pdf_height
            
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            self.gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            print("  Extracting text from page...")
            text_elements = self.extract_text_from_image(page_num)
            print(f"  Found {len(text_elements)} text elements")
            
            matches = self.multi_scale_template_match()
            matches = self.remove_overlapping_detections(matches)
            
            draw_image = pil_image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            symbol_counts = {}
            
            for text in text_elements:
                img_bbox = self.convert_pdf_to_image_coords(text['bbox'], pdf_page, (img_width, img_height))
                x, y, w, h = img_bbox
                draw.rectangle([x, y, x + w, y + h], outline='green', width=2)
            
            for match in matches:
                x, y, w, h = match['bbox']
                confidence = match['confidence']
                name = match['name']
                
                pdf_bbox = self.convert_image_to_pdf_coords((x, y, w, h), pdf_page, (img_width, img_height))
                pdf_x, pdf_y, pdf_w, pdf_h = pdf_bbox
                print(f"    Symbol {name} PDF coords: ({pdf_x:.1f}, {pdf_y:.1f})")
                
                found_text = self.find_best_matching_text(
                    (x, y, w, h), 
                    text_elements, 
                    name, 
                    pdf_page, 
                    (img_width, img_height)
                )
                
                print(f"    Found text: '{found_text}'")
                
                if found_text:
                    allowed_prefixes = self.label_prefixes.get(name, [])
                    if allowed_prefixes:
                        text_matches = any(found_text.startswith(prefix) for prefix in allowed_prefixes)
                        if text_matches:
                            label = found_text
                        else:
                            label = "Default Name"
                            failed_prefix_log.append({
                                'page': page_num + 1,
                                'symbol_type': name,
                                'symbol_location': (x, y),
                                'found_text': found_text,
                                'expected_prefixes': allowed_prefixes,
                                'confidence': confidence
                            })
                    else:
                        label = found_text
                else:
                    label = "Default Name"
                
                draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
                display_text = f"{name}({confidence:.2f}) - {label}"
                draw.text((x, y - 15), display_text, fill='red')
                
                symbol_counts[name] = symbol_counts.get(name, 0) + 1
                print(f"  {name}({confidence:.2f}) | label:'{label}' | at ({x},{y})")
                
                match['label'] = label
            
            print(f"  Found {len(matches)} symbols")
            for symbol, count in symbol_counts.items():
                threshold = self.get_symbol_threshold(symbol)
                print(f"    {symbol}: {count} (threshold: {threshold:.2f})")
            
            all_matches_data.append({
                'page': page_num + 1,
                'matches': matches
            })
            
            highlighted_images.append(draw_image)
        
        if highlighted_images:
            highlighted_images[0].save(output_path, "PDF", save_all=True, append_images=highlighted_images[1:])
            print(f"Output saved: {output_path}")
        
        symbols_json_path = output_path.replace('.pdf', '_symbols.json')
        self.save_symbols_to_json(all_matches_data, symbols_json_path)
        
        self.pdf_doc.close()
        
        if failed_prefix_log:
            print(f"\nFailed prefix matches ({len(failed_prefix_log)} total):")
            for entry in failed_prefix_log:
                print(f"  Page {entry['page']}: {entry['symbol_type']} at ({entry['symbol_location'][0]},{entry['symbol_location'][1]})")
                print(f"    Found text: '{entry['found_text']}'")
                print(f"    Expected prefixes: {entry['expected_prefixes']}")
                print(f"    Confidence: {entry['confidence']:.2f}")
            
            failed_log_file = output_path.replace('.pdf', '_failed_prefixes.json')
            with open(failed_log_file, 'w') as f:
                json.dump(failed_prefix_log, f, indent=2, default=str)
            print(f"\nFailed prefix log saved to: {failed_log_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input PDF file')
    parser.add_argument('--output', required=True, help='Output PDF file')
    parser.add_argument('--templates', required=True, help='Template directory')
    parser.add_argument('--thresholds', required=True, help='JSON file with individual symbol thresholds')
    parser.add_argument('--prefixes', required=True, help='JSON file with symbol label prefixes')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    detector = PIDDetector()
    detector.create_highlighted_pdf(args.input, args.output, args.templates, args.thresholds, args.prefixes)

if __name__ == "__main__":
    main()