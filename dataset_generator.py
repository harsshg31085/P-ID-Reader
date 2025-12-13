from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import math
import time
import json
import os
import multiprocessing as mp

class DatasetGenerator:
    def __init__(self, color):
        self.color = color
        self.pipe_color = color
        self.font_cache = {}
        self.class_map = {
            'valve': 1,
            'bell_valve': 2,
            'controller': 3,
            'heat_exchanger': 4,
            'vessel': 5
        }

    def _get_font(self, size):
        font_path = random.choice(['assets/arial.ttf', 'assets/calibri.ttf', 'assets/dejavusans.ttf'])
        key = (font_path, size)
        if key not in self.font_cache:
            self.font_cache[key] = ImageFont.truetype(font_path, size)
        return self.font_cache[key]

    def connect_points(self, img, draw, conn1, conn2, img_width, img_height):
        x0, y0 = conn1
        x1, y1 = conn2

        x0 *= img_width
        y0 *= img_height
        x1 *= img_width
        y1 *= img_height
        
        x0 = max(0, min(x0, img_width))
        y0 = max(0, min(y0, img_height))
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        
        dx = abs(x0 - x1)
        dy = abs(y0 - y1)
        
        arrow_at_symbol1 = random.random() > 0.7
        arrow_at_symbol2 = random.random() > 0.7
        
        use_dashed_line = random.random() > 0.8
        
        if dx < dy:
            mid_y = (y0 + y1) / 2
            mid_y = max(0, min(mid_y, img_height))
            
            if use_dashed_line:
                self.draw_dashed_line(draw, (x0, y0), (x0, mid_y), width=random.randint(1,3))
                self.draw_dashed_line(draw, (x1, y1), (x1, mid_y), width=random.randint(1,3))
                self.draw_dashed_line(draw, (x0, mid_y), (x1, mid_y), width=random.randint(1,3))
            else:
                draw.line((x0, y0, x0, mid_y), fill=self.pipe_color, width=random.randint(1,3))
                draw.line((x1, y1, x1, mid_y), fill=self.pipe_color, width=random.randint(1,3))
                draw.line((x0, mid_y, x1, mid_y), fill=self.pipe_color, width=random.randint(1,3))
            
            if arrow_at_symbol1:
                direction = -1 if y0 < mid_y else 1
                self.draw_arrow_into_symbol(draw, x0, y0, direction, 'vertical')
            
            if arrow_at_symbol2:
                direction = -1 if y1 < mid_y else 1
                self.draw_arrow_into_symbol(draw, x1, y1, direction, 'vertical')
                
        else:
            mid_x = (x0 + x1) / 2
            mid_x = max(0, min(mid_x, img_width))
            
            if use_dashed_line:
                self.draw_dashed_line(draw, (x0, y0), (mid_x, y0), width=random.randint(1,3))
                self.draw_dashed_line(draw, (x1, y1), (mid_x, y1), width=random.randint(1,3))
                self.draw_dashed_line(draw, (mid_x, y0), (mid_x, y1), width=random.randint(1,3))
            else:
                draw.line((x0, y0, mid_x, y0), fill=self.pipe_color, width=random.randint(1,3))
                draw.line((x1, y1, mid_x, y1), fill=self.pipe_color, width=random.randint(1,3))
                draw.line((mid_x, y0, mid_x, y1), fill=self.pipe_color, width=random.randint(1,3))
            
            if arrow_at_symbol1:
                direction = -1 if x0 < mid_x else 1
                self.draw_arrow_into_symbol(draw, x0, y0, direction, 'horizontal')
            
            if arrow_at_symbol2:
                direction = -1 if x1 < mid_x else 1
                self.draw_arrow_into_symbol(draw, x1, y1, direction, 'horizontal')

    def draw_dashed_line(self, draw, start, end, width=3, dash_length=10):
        x1, y1 = start
        x2, y2 = end
        
        dx = x2 - x1
        dy = y2 - y1
        distance = (dx**2 + dy**2)**0.5
        
        if distance == 0:
            return
        
        ux = dx / distance
        uy = dy / distance
        
        total = 0
        while total < distance:
            seg_end = total + dash_length
            if seg_end > distance:
                seg_end = distance
                
            sx1 = x1 + ux * total
            sy1 = y1 + uy * total
            sx2 = x1 + ux * seg_end
            sy2 = y1 + uy * seg_end
            
            draw.line([(sx1, sy1), (sx2, sy2)], fill=self.pipe_color, width=width)
            
            total = seg_end + dash_length

    def draw_arrow_into_symbol(self, draw, x, y, direction, orientation):
        arrow_size = random.randint(6, 12)
        
        if orientation == 'horizontal':
            if direction == 1:
                points = [
                    (x, y),
                    (x - arrow_size, y - arrow_size/2),
                    (x - arrow_size, y + arrow_size/2),
                ]
            else:
                points = [
                    (x, y),
                    (x + arrow_size, y - arrow_size/2),
                    (x + arrow_size, y + arrow_size/2),
                ]
        else:
            if direction == 1:
                points = [
                    (x, y),
                    (x - arrow_size/2, y - arrow_size),
                    (x + arrow_size/2, y - arrow_size),
                ]
            else:
                points = [
                    (x, y),
                    (x - arrow_size/2, y + arrow_size),
                    (x + arrow_size/2, y + arrow_size),
                ]
        
        draw.polygon(points, fill=self.pipe_color, outline=self.pipe_color)

    def generate_valve(self, img, draw, x_norm, y_norm, w_norm, h_norm, orientation=None):
        W, H = img.size
        x0, y0 = W * x_norm, H * y_norm
        width, height = w_norm * W, h_norm * H
        x1, y1 = x0 + width, y0 + height

        draw_horizontal = width > height if orientation is None else (1 - orientation)
    
        if draw_horizontal:
            draw.line((x0, y0, x1, y1), fill=self.color, width=random.randint(1,3))
            draw.line((x1, y1, x1, y0), fill=self.color, width=random.randint(1,3))
            draw.line((x1, y0, x0, y1), fill=self.color, width=random.randint(1,3))
            draw.line((x0, y1, x0, y0), fill=self.color, width=random.randint(1,3))
            
            visual_data = {
                'type': 'valve',
                'right_connection': (x_norm + w_norm, y_norm + h_norm/2),
                'left_connection': (x_norm, y_norm + h_norm/2),
                'top_connection': (x_norm + w_norm/2, y_norm + h_norm/2),
                'bottom_connection': (x_norm + w_norm/2, y_norm + h_norm/2),
                'position': (x_norm, y_norm),
                'size': (w_norm, h_norm),
                'grid_position': None,  
                'is_horizontal': True
            }
        else:
            draw.line((x0, y0, x1, y0), fill=self.color, width=random.randint(1,3))
            draw.line((x1, y0, x0, y1), fill=self.color, width=random.randint(1,3))
            draw.line((x0, y1, x1, y1), fill=self.color, width=random.randint(1,3))
            draw.line((x1, y1, x0, y0), fill=self.color, width=random.randint(1,3))
            
            visual_data = {
                'type': 'valve',
                'right_connection': (x_norm + w_norm/2, y_norm + h_norm/2),
                'left_connection': (x_norm + w_norm/2, y_norm + h_norm/2),
                'top_connection': (x_norm + w_norm/2, y_norm),
                'bottom_connection': (x_norm + w_norm/2, y_norm + h_norm),
                'position': (x_norm, y_norm),
                'size': (w_norm, h_norm),
                'grid_position': None,  
                'is_horizontal': False
            }
        
        annotation = {
            'bbox': [round(float(x0), 2), round(float(y0), 2), round(float(width), 2), round(float(height), 2)],
            'category_id': self.class_map['valve'],
            'category_name': 'valve'
        }
        
        return visual_data, annotation

    def generate_bell_valve(self, img, draw, x_norm, y_norm, w_norm, h_norm):
        W, H = img.size
        x0, y0 = W * x_norm, H * y_norm
        width, height = w_norm * W, h_norm * H
        x1, y1 = x0 + width, y0 + height

        R = min(width, height) / 3
        stick_height = R * random.uniform(0.9, 1.2)

        if width > height:
            center_x = (x0 + x1)/2

            if random.random() > 0.5:
                center_y = y0 + R + stick_height
                draw.line((center_x, center_y, center_x, center_y - stick_height), fill=self.color, width=random.randint(1,3))
                draw.pieslice((center_x - R, center_y - stick_height - R, center_x + R, center_y - stick_height + R), start=180, end=360, outline=self.color, width=random.randint(1,3), fill=None)
                self.generate_valve(img, draw, x_norm, (2*center_y - y1)/H, w_norm, 2*(y1 - center_y)/H, orientation=0)

                visual_data = {
                    'type': 'bell_valve',
                    'orientation': 'horizontal',
                    'bottom_connection': (center_x/W, center_y/H),
                    'top_connection': (center_x/W, (center_y - stick_height - R)/H),
                    'left_connection': (x0/W, center_y/H),
                    'right_connection': (x1/W, center_y/H),
                    'grid_position': None
                }
            else:
                center_y = y1 - R - stick_height
                draw.line((center_x, center_y, center_x, center_y + stick_height), fill=self.color, width=random.randint(1,3))
                draw.pieslice((center_x - R, center_y + stick_height - R, center_x + R, center_y + stick_height + R), start=0, end=180, outline=self.color, fill=None, width=random.randint(1,3))
                self.generate_valve(img, draw, x_norm, y_norm, w_norm, 2*(center_y - y0)/H, orientation=0)

                visual_data = {
                    'type': 'bell_valve',
                    'orientation': 'horizontal',
                    'bottom_connection': (center_x/W, (center_y + stick_height + R)/H),
                    'top_connection': (center_x/W, center_y/H),
                    'right_connection': (x1/W, center_y/H),
                    'left_connection': (x0/W, center_y/H),
                    'grid_position': None
                }
        else:
            center_y = (y0 + y1)/2

            if random.random() > 0.5:
                center_x = x1 - R - stick_height
                draw.line((center_x, center_y, center_x + stick_height, center_y), fill=self.color, width=random.randint(1,3))
                draw.pieslice((center_x + stick_height - R, center_y - R, center_x + stick_height + R, center_y + R), start=270, end=90, outline=self.color, fill=None, width=random.randint(1,3))
                self.generate_valve(img, draw, x_norm, y_norm, 2*(center_x - x0)/H, h_norm, orientation=1)

                visual_data = {
                    'type': 'bell_valve',
                    'orientation': 'vertical',
                    'bottom_connection': (center_x/W, y1/H),
                    'top_connection': (center_x/W, y0/H),
                    'right_connection': ((center_x + stick_height + R)/W, center_y/H),
                    'left_connection': (center_x/W, center_y/H)
                }
            else:
                center_x = x0 + R + stick_height
                draw.line((center_x, center_y, center_x - stick_height, center_y), fill=self.color, width=random.randint(1,3))
                draw.pieslice((center_x - stick_height - R, center_y - R, center_x - stick_height + R, center_y + R), start=90, end=270, outline=self.color, fill=None, width=random.randint(1,3))
                self.generate_valve(img, draw, (2*center_x - x1)/W, y_norm, 2*(x1 - center_x)/W, h_norm, orientation=1)

                visual_data = {
                    'type': 'bell_valve',
                    'orientation': 'vertical',
                    'bottom_connection': (center_x/W, y1/H),
                    'top_connection': (center_x/W, y0/H),
                    'right_connection': (center_x/W, center_y/H),
                    'left_connection': ((center_x - R - stick_height)/W, center_y/H)
                }
        
        annotation = {
            'bbox': [round(float(x0), 2), round(float(y0), 2), round(float(width), 2), round(float(height), 2)],
            'category_id': self.class_map['bell_valve'],
            'category_name': 'bell_valve'
        }
        
        return visual_data, annotation

    def generate_heat_exchanger(self, img, draw, x_norm, y_norm, r_norm):
        W, H = img.size
        radius = r_norm * H
        x0, y0 = x_norm * W, y_norm * H
        center_x, center_y = x0 + radius, y0 + radius
        x1, y1 = x0 + 2 * radius, y0 + 2 * radius

        draw.ellipse((x0, y0, x1, y0 + 2 * radius), outline=self.color, width=random.randint(1,3))

        orientation = random.randint(2, 3)
        p1, p2, h1 = random.uniform(0,0.3), random.uniform(0.55,0.75), random.uniform(0.4,0.6)

        if orientation == 0:
            draw.line((x0, center_y, x0 + p1 * radius, center_y), fill=self.color, width=random.randint(1,3))
            draw.line((x1, center_y, x1 - p1 * radius, center_y), fill=self.color, width=random.randint(1,3))
            draw.line((x0 + p1 * radius, center_y, x0 + p2 * radius, center_y - h1 * radius), fill=self.color, width=random.randint(1,3))
            draw.line((x1 - p1 * radius, center_y, x1 - p2 * radius, center_y + h1 * radius), fill=self.color, width=random.randint(1,3))
            draw.line((x0 + p2 * radius, center_y - h1 * radius, x1 - p2 * radius, center_y + h1 * radius), fill=self.color, width=random.randint(1,3))
        elif orientation == 1:
            draw.line((x0, center_y, x0 + p1 * radius, center_y), fill=self.color, width=random.randint(1,3))
            draw.line((x1, center_y, x1 - p1 * radius, center_y), fill=self.color, width=random.randint(1,3))
            draw.line((x0 + p1 * radius, center_y, x0 + p2 * radius, center_y + h1 * radius), fill=self.color, width=random.randint(1,3))
            draw.line((x1 - p1 * radius, center_y, x1 - p2 * radius, center_y - h1 * radius), fill=self.color, width=random.randint(1,3))
            draw.line((x0 + p2 * radius, center_y + h1 * radius, x1 - p2 * radius, center_y - h1 * radius), fill=self.color, width=random.randint(1,3))
        elif orientation == 2:
            draw.line((center_x, y0, center_x, y0 + p1*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x, y1, center_x, y1 - p1*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x, y0 + p1*radius, center_x + h1*radius, y0 + p2*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x + h1*radius, y0 + p2*radius, center_x - h1*radius, y1 - p2*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x - h1*radius, y1 - p2*radius, center_x, y1 - p1*radius), fill=self.color, width=random.randint(1,3))
        elif orientation == 3:
            draw.line((center_x, y0, center_x, y0 + p1*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x, y1, center_x, y1 - p1*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x, y0 + p1*radius, center_x - h1*radius, y0 + p2*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x - h1*radius, y0 + p2*radius, center_x + h1*radius, y1 - p2*radius), fill=self.color, width=random.randint(1,3))
            draw.line((center_x + h1*radius, y1 - p2*radius, center_x, y1 - p1*radius), fill=self.color, width=random.randint(1,3))
        
        visual_data = {
            'type': 'heat_exchanger',
            'orientation': 'horizontal',
            'bottom_connection': (x_norm + r_norm, y_norm + 2*r_norm),
            'top_connection': (x_norm + r_norm, y_norm),
            'right_connection': (x_norm + 2*r_norm, y_norm + r_norm),
            'left_connection': (x_norm, y_norm + r_norm),
            'position': (x_norm, y_norm),
            'size': (2*r_norm, 2*r_norm),
            'grid_position': None  
        }
        
        annotation = {
            'bbox': [round(float(x0), 2), round(float(y0), 2), round(float(2*radius), 2), round(float(2*radius), 2)],
            'category_id': self.class_map['heat_exchanger'],
            'category_name': 'heat_exchanger'
        }
        
        return visual_data, annotation

    def generate_controller(self, img, draw, x_norm, y_norm, r_norm):
        W, H = img.size
        x0, y0 = x_norm * W, y_norm * H
        radius = r_norm * H

        name = random.choice(["PC", "HC", "FC", "TC", "PDC", "LC", "AC", "SC", "HS", "PIC", "HIC", "FIC", "TIC", "PIDC", "LIC", "AIC", "SIC", "HIS"])
        type = random.randint(0, 1)

        if type == 0:
            draw.ellipse((x0, y0, x0 + 2*radius, y0 + 2*radius), fill=None, outline=self.color, width=random.randint(1,3))
            x1, y1, x2, y2 = x0 + 0.3*radius, y0 + 0.7*radius, x0 + 1.7*radius, y0 + 1.3*radius
            font = self._fit_text_to_box(draw, name, x2-x1, y2-y1)
            self._draw_centered_text(draw, name, x1, y1, x2-x1, y2-y1, font)
        if type == 1:
            draw.ellipse((x0, y0, x0 + 2*radius, y0 + 2*radius), fill=None, outline=self.color, width=random.randint(1,3))
            draw.rectangle((x0, y0, x0 + 2*radius, y0 + 2*radius), width=random.randint(1,3), outline=self.color)
            draw.line((x0, y0 + radius, x0 + 2*radius, y0 + radius), fill=self.color, width=random.randint(1,3))
            x1, y1, x2, y2 = x0 + 0.4*radius, y0 + 0.3*radius, x0 + 1.6*radius, y0 + 0.9*radius
            font = self._fit_text_to_box(draw, name, x2-x1, y2-y1)
            self._draw_centered_text(draw, name, x1, y1, x2-x1, y2-y1, font)

        visual_data = {
            'type': 'controller',
            'label': name,
            'controller_type': type,
            'right_connection': (x_norm + 2*r_norm, y_norm + r_norm),
            'left_connection': (x_norm, y_norm + r_norm),
            'top_connection': (x_norm + r_norm, y_norm),
            'bottom_connection': (x_norm + r_norm, y_norm + 2*r_norm),
            'position': (x_norm, y_norm),
            'size': (2*r_norm, 2*r_norm),
            'grid_position': None  
        }
        
        annotation = {
            'bbox': [round(float(x0), 2), round(float(y0), 2), round(float(2*radius), 2), round(float(2*radius), 2)],
            'category_id': self.class_map['controller'],
            'category_name': 'controller',
            'label': name
        }
        
        return visual_data, annotation

    def generate_vessel(self, img, draw, x_norm, y_norm, w_norm, h_norm):
        W, H = img.size
        x0, y0 = x_norm * W, H * y_norm
        width, height = w_norm * W, h_norm * H

        if height > width:
            R = random.uniform(0.5, 2.0) * width
            ratio = width / (2 * R)
            if abs(ratio) > 1:
                ratio = 1.0 if ratio > 0 else -1.0
            theta = math.asin(ratio)
            deg = math.degrees(theta)
            cos = math.cos(theta)

            y1 = y0 + R * (1 - cos)
            y2 = y0 + height - R * (1 - cos)

            draw.line((x0, y1, x0, y2), fill=self.color, width=random.randint(1,3))
            draw.line((x0 + width, y1, x0 + width, y2), fill=self.color, width=random.randint(1,3))

            if random.random() > 0.5:
                draw.line((x0, y1, x0 + width, y1), fill=self.color, width=random.randint(1,3))
                draw.line((x0, y2, x0 + width, y2), fill=self.color, width=random.randint(1,3))

            xm = x0 + width / 2
            draw.arc((xm - R, y0, xm + R, y0 + 2 * R), fill=self.color, width=random.randint(1,3), start=270 - deg, end=270 + deg)
            draw.arc((xm - R, y0 + height - 2 * R, xm + R, y0 + height), fill=self.color, width=random.randint(1,3), start=90 - deg, end=90 + deg)
            
            left_conn_y = random.uniform(y1/H, y2/H)
            right_conn_y = random.uniform(y1/H, y2/H)
            
            visual_data = {
                'type': 'vessel',
                'orientation': 'vertical',
                'top_connection': (x_norm + w_norm/2, y_norm),
                'bottom_connection': (x_norm + w_norm/2, y_norm + h_norm),
                'right_connection': (x_norm + w_norm, right_conn_y),
                'left_connection': (x_norm, left_conn_y),
                'position': (x_norm, y_norm),
                'size': (w_norm, h_norm),
                'radius_ratio': R/width,
                'grid_position': None  
            }
        else:
            R = random.uniform(0.5, 2.0) * height
            ratio = height / (2 * R)
            if abs(ratio) > 1:
                ratio = 1.0 if ratio > 0 else -1.0
            theta = math.asin(ratio)
            deg = math.degrees(theta)
            cos = math.cos(theta)

            x1 = x0 + R * (1 - cos)
            x2 = x0 + width - R * (1 - cos)

            draw.line((x1, y0, x2, y0), fill=self.color, width=random.randint(1,3))
            draw.line((x1, y0 + height, x2, y0 + height), fill=self.color, width=random.randint(1,3))

            if random.random() > 0.5:
                draw.line((x1, y0, x1, y0 + height), fill=self.color, width=random.randint(1,3))
                draw.line((x2, y0, x2, y0 + height), fill=self.color, width=random.randint(1,3))

            ym = y0 + height / 2
            draw.arc((x0, ym - R, x0 + 2 * R, ym + R), fill=self.color, width=random.randint(1,3), start=180 - deg, end=180 + deg)
            draw.arc((x0 + width - 2 * R, ym - R, x0 + width, ym + R), fill=self.color, width=random.randint(1,3), start=360 - deg, end=360 + deg)
            
            top_conn_x = random.uniform(x1, x2)
            bottom_conn_x = random.uniform(x1, x2)
            
            visual_data = {
                'type': 'vessel',
                'orientation': 'horizontal',
                'top_connection': (top_conn_x, y_norm),
                'bottom_connection': (bottom_conn_x, y_norm + h_norm),
                'right_connection': (x_norm + w_norm, y_norm + h_norm/2),
                'left_connection': (x_norm, y_norm + h_norm/2),
                'position': (x_norm, y_norm),
                'size': (w_norm, h_norm),
                'radius_ratio': R/height,
                'grid_position': None  
            }
        
        annotation = {
            'bbox': [round(float(x0), 2), round(float(y0), 2), round(float(width), 2), round(float(height), 2)],
            'category_id': self.class_map['vessel'],
            'category_name': 'vessel'
        }
        
        return visual_data, annotation

    def _fit_text_to_box(self, draw, text, box_w, box_h):
        low, high = 1, 600
        best_font = None

        while low <= high:
            mid = (low + high) // 2
            font = self._get_font(mid)

            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            if tw <= box_w and th <= box_h:
                best_font = font
                low = mid + 1
            else:
                high = mid - 1

        return best_font

    def _draw_centered_text(self, draw, text, x, y, w, h, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tx = x + (w - tw) / 2 - bbox[0]
        ty = y + (h - th) / 2 - bbox[1]
        draw.text((tx, ty), text, fill="black", font=font)

    def _valid_box(self, bbox, bounding_boxes, padding=0.02):
        if not bbox: return False
        def is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2):
            return not (
                x1 + w1 + padding < x2 or  
                x2 + w2 + padding < x1 or  
                y1 + h1 + padding < y2 or
                y2 + h2 + padding < y1
            )
        
        for box in bounding_boxes:
            if is_overlapping(*bbox, *box): return False
        return True

    def _generate_bounding_box_grid(self, type, img_width, img_height, grid_pos):
        grid_x, grid_y = grid_pos
        cell_width = img_width / 8
        cell_height = img_height / 8
        
        x1, y1 = self._randomized_grid_center(grid_x, grid_y, img_width, img_height)
        
        parameters = {
            'heat_exchanger': (random.uniform(0.02, 0.03), None),
            'controller': (random.uniform(0.0075, 0.015), None),
            'valve': (random.uniform(0.015, 0.02), random.uniform(1.5, 2.0)),
            'vessel': (random.uniform(0.06, 0.09), random.uniform(1.5, 3.5)),
            'bell_valve': (random.uniform(0.015, 0.02), random.uniform(1.5, 2.0))
        }

        size, ratio = parameters[type]
        if not ratio:
            r = size * img_height
            x1 = x1 - r
            y1 = y1 - r
            if x1 < 0 or x1 + 2*r > img_width or y1 < 0 or y1 + 2*r > img_height: 
                return None
            return (x1/img_width, y1/img_height, r/img_height)
        else:
            if random.random() > 0.5:
                w = size * img_width
                h = ratio * w
            else:
                h = size * img_height
                w = ratio * h
            
            x1 = x1 - w/2
            y1 = y1 - h/2
            
            if x1 < 0 or x1 + w > img_width or y1 < 0 or y1 + h > img_height: 
                return None
            return (x1/img_width, y1/img_height, w/img_width, h/img_height)

    def find_connecting_neighbors(self, symbol_grid, grid_x, grid_y, grid_width, grid_height):
        symbol = symbol_grid[grid_y][grid_x]
        if not symbol:
            return []
        
        neighbors = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)] 
        
        for dx, dy in directions:
            nx, ny = grid_x + dx, grid_y + dy
            
            while 0 <= nx < grid_width and 0 <= ny < grid_height:
                neighbor = symbol_grid[ny][nx]
                if neighbor:
                    neighbors.append((neighbor, (dx, dy)))
                    break
                nx += dx
                ny += dy
        
        return neighbors

    def process_connections(self, img, draw, symbol_grid, grid_width, grid_height, img_width, img_height):
        processed_pairs = set()
        
        for y in range(grid_height):
            for x in range(grid_width):
                symbol = symbol_grid[y][x]
                if not symbol:
                    continue
                    
                neighbors = self.find_connecting_neighbors(symbol_grid, x, y, grid_width, grid_height)
                
                for neighbor, (dx, dy) in neighbors:
                    neighbor_pos = None
                    for ny in range(grid_height):
                        for nx in range(grid_width):
                            if symbol_grid[ny][nx] is neighbor:
                                neighbor_pos = (nx, ny)
                                break
                        if neighbor_pos:
                            break
                    
                    if not neighbor_pos:
                        continue
                    
                    pair_id = frozenset([(x, y), neighbor_pos])
                    if pair_id in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair_id)
                    
                    conn1 = None
                    conn2 = None
                    
                    if dx == 1: 
                        conn1 = symbol.get('right_connection')
                        conn2 = neighbor.get('left_connection')
                    elif dx == -1:  
                        conn1 = symbol.get('left_connection')
                        conn2 = neighbor.get('right_connection')
                    elif dy == 1:  
                        conn1 = symbol.get('bottom_connection')
                        conn2 = neighbor.get('top_connection')
                    elif dy == -1:  
                        conn1 = symbol.get('top_connection')
                        conn2 = neighbor.get('bottom_connection')
                    
                    if not conn1 or not conn2:
                        continue
                    
                    if (conn1[0] < 0 or conn1[0] > 1 or conn1[1] < 0 or conn1[1] > 1 or
                        conn2[0] < 0 or conn2[0] > 1 or conn2[1] < 0 or conn2[1] > 1):
                        continue
                    
                    self.connect_points(img, draw, conn1, conn2, img_width, img_height)

    def generate_diagram(self, index, dataset_type='train'):
        line_alpha = random.randint(220, 255)
        self.color = (0, 0, 0, line_alpha)
        self.pipe_color = (0, 0, 0, line_alpha)

        W, H = 1536, 1536
        
        os.makedirs(f'./data/{dataset_type}/images', exist_ok=True)
        os.makedirs(f'./data/{dataset_type}/labels', exist_ok=True)
        
        img = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(img)
        
        image_filename = f'./data/{dataset_type}/images/{index}.png'
        json_filename = f'./data/{dataset_type}/labels/{index}.json'

        grid_width = 8 
        grid_height = 8 
            
        symbol_grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
            
        grid_cells = [(x, y) for x in range(grid_width) for y in range(grid_height)]
        random.shuffle(grid_cells)
            
        symbols = []
        grid_positions = []
            
        symbol_counts = {
            'heat_exchanger': random.randint(2, 3),
            'controller': random.randint(20, 30),  
            'valve': random.randint(5, 6),
            'vessel': random.randint(1, 2),
            'bell_valve': random.randint(5, 6)
        }
            
        for symbol_type, count in symbol_counts.items():
            for _ in range(count):
                if grid_cells:
                    symbols.append(symbol_type)
                    grid_positions.append(grid_cells.pop())
                else:
                    break

        bounding_boxes = []
        annotations = []
        annotation_id_counter = 0
        
        for symbol_type, grid_pos in zip(symbols, grid_positions):
            bbox = None
            for _ in range(1000):  
                cand = self._generate_bounding_box_grid(symbol_type, W, H, grid_pos)
                if cand and self._valid_box(
                    self._get_bbox_from_cand(cand, symbol_type, W, H), 
                    bounding_boxes, padding=0.03):
                    bbox = cand
                    break

            if not bbox:
                continue

            symbol_visual = None
            symbol_annotation = None
            
            if symbol_type == 'heat_exchanger':
                symbol_visual, symbol_annotation = self.generate_heat_exchanger(img, draw, *bbox)
            elif symbol_type == 'controller':
                symbol_visual, symbol_annotation = self.generate_controller(img, draw, *bbox)
            elif symbol_type == 'valve':
                symbol_visual, symbol_annotation = self.generate_valve(img, draw, *bbox)
            elif symbol_type == 'bell_valve':
                symbol_visual, symbol_annotation = self.generate_bell_valve(img, draw, *bbox)
            elif symbol_type == 'vessel':
                symbol_visual, symbol_annotation = self.generate_vessel(img, draw, *bbox)

            if symbol_visual:
                grid_x, grid_y = grid_pos
                symbol_visual['grid_position'] = (grid_x, grid_y)
                symbol_grid[grid_y][grid_x] = symbol_visual
                
                bbox_pixel = symbol_annotation['bbox']
                annotation = {
                    'id': annotation_id_counter,
                    'image_id': index,
                    'category_id': symbol_annotation['category_id'],
                    'bbox': bbox_pixel,
                    'area': round(bbox_pixel[2] * bbox_pixel[3], 2),
                    'iscrowd': 0,
                    'segmentation': []
                }
                annotations.append(annotation)
                annotation_id_counter += 1

                bbox_coords = self._get_bbox_from_cand(bbox, symbol_type, W, H)
                bounding_boxes.append(bbox_coords)

        self.process_connections(img, draw, symbol_grid, grid_width, grid_height, W, H)

        img.save(image_filename)
        
        coco_json = {
            'info': {
                'description': 'P&ID Diagram Dataset',
                'version': '1.0'
            },
            'licenses': [],
            'categories': [
                {'id': 1, 'name': 'valve', 'supercategory': 'symbol'},
                {'id': 2, 'name': 'bell_valve', 'supercategory': 'symbol'},
                {'id': 3, 'name': 'controller', 'supercategory': 'symbol'},
                {'id': 4, 'name': 'heat_exchanger', 'supercategory': 'symbol'},
                {'id': 5, 'name': 'vessel', 'supercategory': 'symbol'}
            ],
            'images': [{
                'id': index,
                'file_name': f'{index}.png',
                'width': W,
                'height': H
            }],
            'annotations': annotations
        }
        
        with open(json_filename, 'w') as f:
            json.dump(coco_json, f, indent=2)
        
        print(f"Generated diagram {index} for {dataset_type}")
        
        return coco_json

    def _randomized_grid_center(self, grid_x, grid_y, img_width, img_height, max_offset=0.18):
        cell_w = img_width / 8
        cell_h = img_height / 8

        base_x = grid_x * cell_w + cell_w * 0.5
        base_y = grid_y * cell_h + cell_h * 0.5

        ox = random.uniform(-max_offset, max_offset) * cell_w
        oy = random.uniform(-max_offset, max_offset) * cell_h

        final_x = min(max(base_x + ox, grid_x * cell_w + cell_w * 0.18),
                    (grid_x + 1) * cell_w - cell_w * 0.18)
        final_y = min(max(base_y + oy, grid_y * cell_h + cell_h * 0.18),
                    (grid_y + 1) * cell_h - cell_h * 0.18)

        return final_x, final_y

    def _get_bbox_from_cand(self, cand, symbol_type, W, H):
        if symbol_type in ['heat_exchanger', 'controller']:
            x, y, r = cand
            return (x * W, y * H, 2 * r * H, 2 * r * H)
        else:  
            x, y, w, h = cand
            return (x * W, y * H, w * W, h * H)

def worker(args):
    i, dataset_type = args
    gen = DatasetGenerator('black')
    
    annotation = gen.generate_diagram(i, dataset_type)
    
    return annotation

def generate_dataset(total_images=10000):    
    base_dir = './data/'
    splits = ['train', 'val', 'test']
    
    for split in splits:
        os.makedirs(f'{base_dir}{split}/images', exist_ok=True)
        os.makedirs(f'{base_dir}{split}/labels', exist_ok=True)
    
    train_count = int(total_images * 0.8)
    val_count = int(total_images * 0.1)
    
    args_list = []
    for i in range(total_images):
        if i < train_count:
            dataset_type = 'train'
        elif i < train_count + val_count:
            dataset_type = 'val'
        else:
            dataset_type = 'test'
        args_list.append((i, dataset_type))
    
    print(f"Generating {total_images} images with 8:1:1 split...")
    print(f"Split: {train_count} train, {val_count} val, {total_images - train_count - val_count} test")
    
    start = time.perf_counter()
    
    num_workers = max(1, mp.cpu_count() - 1)
    
    with mp.Pool(num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(worker, args_list)):
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{total_images} images")
    
    print(f'Computation time: {time.perf_counter() - start} seconds')

if __name__ == '__main__':
    generate_dataset(total_images=10)
