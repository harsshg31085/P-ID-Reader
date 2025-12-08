from PIL import Image, ImageDraw, ImageFont
import random
import math
import time

class DatasetGenerator:
    def __init__(self, color):
        self.color = color
        self.font_cache = {}

    def _get_font(self, size, font_path="data/arial.ttf"):
        key = (font_path, size)
        if key not in self.font_cache:
            self.font_cache[key] = ImageFont.truetype(font_path, size)
        return self.font_cache[key]

    def generate_valve(self, img, draw, x_norm, y_norm, w_norm, h_norm):
        W, H = img.size
        x0, y0 = W * x_norm, H * y_norm
        width, height = w_norm * W, h_norm * H
        x1, y1 = x0 + width, y0 + height

        if width > height:
            draw.line((x0, y0, x1, y1), fill=self.color, width=2)
            draw.line((x1, y1, x1, y0), fill=self.color, width=2)
            draw.line((x1, y0, x0, y1), fill=self.color, width=2)
            draw.line((x0, y1, x0, y0), fill=self.color, width=2)
        else:
            draw.line((x0, y0, x1, y0), fill=self.color, width=2)
            draw.line((x1, y0, x0, y1), fill=self.color, width=2)
            draw.line((x0, y1, x1, y1), fill=self.color, width=2)
            draw.line((x1, y1, x0, y0), fill=self.color, width=2)

    def generate_bell_valve(self, img, draw, x_norm, y_norm, w_norm, h_norm, v_orientation=0, h_orientation=0):
        W, H = img.size

        x0, y0 = W * x_norm, H * y_norm
        width, height = w_norm * W, h_norm * H
        x1, y1 = x0 + width, y0 + height

        R = min(width, height) / 3
        stick_height = R / 2

        valve_height = height - (R + stick_height)
        valve_y0 = y0 + (0 if h_orientation else R + stick_height)

        self.generate_valve(
            img, draw,
            x_norm,
            valve_y0 / H,
            w_norm,
            valve_height / H
        )

        xm = (x0 + x1) / 2
        ym = valve_y0 + valve_height / 2

        # Stick + bell
        if v_orientation == 0:
            if h_orientation == 0:
                stick_end_y = y0
                draw.line((xm, ym, xm, stick_end_y), fill=self.color, width=2)
                draw.pieslice(
                    (xm - R, stick_end_y - 2 * R, xm + R, stick_end_y),
                    start=180, end=360, fill=None, outline=self.color, width=2
                )
            else:  # bell BELOW
                stick_end_y = y1
                draw.line((xm, ym, xm, stick_end_y), fill=self.color, width=2)
                draw.pieslice(
                    (xm - R, stick_end_y, xm + R, stick_end_y + 2 * R),
                    start=0, end=180, fill=None, outline=self.color, width=2
                )

        else:
            if h_orientation == 0:  # bell RIGHT
                stick_end_x = x1
                draw.line((xm, ym, stick_end_x, ym), fill=self.color, width=2)
                draw.pieslice(
                    (stick_end_x, ym - R, stick_end_x + 2 * R, ym + R),
                    start=270, end=90, fill=None, outline=self.color, width=2
                )
            else:  # bell LEFT
                stick_end_x = x0
                draw.line((xm, ym, stick_end_x, ym), fill=self.color, width=2)
                draw.pieslice(
                    (stick_end_x - 2 * R, ym - R, stick_end_x, ym + R),
                    start=90, end=270, fill=None, outline=self.color, width=2
                )

    def generate_heat_exchanger(self, img, draw, x_norm, y_norm, r_norm):
        W, H = img.size
        radius = r_norm * H
        x0, y0 = x_norm * W, y_norm * H
        center_y = y0 + radius
        x1 = x0 + 2 * radius

        draw.ellipse((x0, y0, x1, y0 + 2 * radius), outline=self.color, width=2)

        orientation = random.randint(0, 1)
        p1, p2, h1 = 0.2, 0.6, 0.5

        draw.line((x0, center_y, x0 + p1 * radius, center_y), fill=self.color, width=2)
        draw.line((x1, center_y, x1 - p1 * radius, center_y), fill=self.color, width=2)

        if orientation == 0:
            draw.line((x0 + p1 * radius, center_y, x0 + p2 * radius, center_y - h1 * radius), fill=self.color, width=2)
            draw.line((x1 - p1 * radius, center_y, x1 - p2 * radius, center_y + h1 * radius), fill=self.color, width=2)
            draw.line((x0 + p2 * radius, center_y - h1 * radius, x1 - p2 * radius, center_y + h1 * radius), fill=self.color, width=2)
        else:
            draw.line((x0 + p1 * radius, center_y, x0 + p2 * radius, center_y + h1 * radius), fill=self.color, width=2)
            draw.line((x1 - p1 * radius, center_y, x1 - p2 * radius, center_y - h1 * radius), fill=self.color, width=2)
            draw.line((x0 + p2 * radius, center_y + h1 * radius, x1 - p2 * radius, center_y - h1 * radius), fill=self.color, width=2)

    def generate_controller(self, img, draw, x_norm, y_norm, r_norm):
        W, H = img.size

        x0, y0 = x_norm * W, y_norm * H
        radius = r_norm * H

        name: str = random.choice(["PC", "HC", "FC", "TC", "PDC", "LC", "AC", "SC", "HS", "PIC", "HIC", "FIC", "TIC", "PIDC", "LIC", "AIC", "SIC", "HIS"])
        type = 1

        if type == 0:
            draw.ellipse((x0, y0, x0 + 2*radius, y0 + 2*radius), fill = None, outline = self.color, width = 2)
            x1, y1, x2, y2 = x0 + 0.3*radius, y0 + 0.7*radius, x0 + 1.7*radius, y0 + 1.3*radius

            font = self._fit_text_to_box(draw, name, x2-x1, y2-y1)
            self._draw_centered_text(draw, name, x1, y1, x2-x1, y2-y1, font)
        if type == 1:
            draw.ellipse((x0, y0, x0 + 2*radius, y0 + 2*radius), fill = None, outline = self.color, width = 2)
            draw.rectangle((x0, y0, x0 + 2*radius, y0 + 2*radius), width = 2, outline = self.color)
            draw.line((x0, y0 + radius, x0 + 2*radius, y0 + radius), fill= self.color, width = 2)

            x1, y1, x2, y2 = x0 + 0.4*radius, y0 + 0.3*radius, x0 + 1.6*radius, y0 + 0.9*radius
                
            font = self._fit_text_to_box(draw, name, x2-x1, y2-y1)
            self._draw_centered_text(draw, name, x1, y1, x2-x1, y2-y1, font)

    def generate_vessel(self, img, draw, x_norm, y_norm, w_norm, h_norm):
        W, H = img.size
        x0, y0 = x_norm * W, H * y_norm
        width, height = w_norm * W, h_norm * H

        if height > width:
            R = random.uniform(0.5, 2.0) * width
            theta = math.asin(width / (2 * R))
            deg = math.degrees(theta)
            cos = math.cos(theta)

            y1 = y0 + R * (1 - cos)
            y2 = y0 + height - R * (1 - cos)

            draw.line((x0, y1, x0, y2), fill=self.color, width=2)
            draw.line((x0 + width, y1, x0 + width, y2), fill=self.color, width=2)

            if random.random() > 0.5:
                draw.line((x0, y1, x0 + width, y1), fill=self.color, width=2)
                draw.line((x0, y2, x0 + width, y2), fill=self.color, width=2)

            xm = x0 + width / 2
            draw.arc((xm - R, y0, xm + R, y0 + 2 * R), fill=self.color, width=2, start=270 - deg, end=270 + deg)
            draw.arc((xm - R, y0 + height - 2 * R, xm + R, y0 + height), fill=self.color, width=2, start=90 - deg, end=90 + deg)
        else:
            R = random.uniform(0.5, 2.0) * height
            theta = math.asin(height / (2 * R))
            deg = math.degrees(theta)
            cos = math.cos(theta)

            x1 = x0 + R * (1 - cos)
            x2 = x0 + width - R * (1 - cos)

            draw.line((x1, y0, x2, y0), fill=self.color, width=2)
            draw.line((x1, y0 + height, x2, y0 + height), fill=self.color, width=2)

            if random.random() > 0.5:
                draw.line((x1, y0, x1, y0 + height), fill=self.color, width=2)
                draw.line((x2, y0, x2, y0 + height), fill=self.color, width=2)

            ym = y0 + height / 2
            draw.arc((x0, ym - R, x0 + 2 * R, ym + R), fill=self.color, width=2, start=180 - deg, end=180 + deg)
            draw.arc((x0 + width - 2 * R, ym - R, x0 + width, ym + R), fill=self.color, width=2, start=360 - deg, end=360 + deg)

    def _fit_text_to_box(self, draw, text, box_w, box_h, font_path="data/arial.ttf"):
        low, high = 1, 600
        best_font = None

        while low <= high:
            mid = (low + high) // 2
            font = self._get_font(mid, font_path)

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

    def _valid_box(self, bbox, bounding_boxes):
        if not bbox: return False
        def is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2):
            return not (
                x1 + w1 < x2 or  
                x2 + w2 < x1 or  
                y1 + h1 < y2 or
                y2 + h2 < y1
            )
        
        for box in bounding_boxes:
            if is_overlapping(*bbox, *box): return False
        return True

    def _generate_bounding_box(self, type, img_width, img_height):
        x1 = random.uniform(0.05, 0.95) * img_width
        y1 = random.uniform(0.05, 0.95) * img_height

        parameters = {
            'heat_exchanger': (random.uniform(0.01, 0.02), None),
            'controller': (random.uniform(0.01, 0.015), None),
            'valve': (random.uniform(0.01, 0.015), random.uniform(2,2.5)),
            #'bell_valve': (random.uniform(0.01, 0.025), random.uniform(2,2.5)),
            'vessel': (random.uniform(0.05, 0.075), random.uniform(2,2.5))
        }

        size, ratio = parameters[type]
        if not ratio:
            r = size * img_height
            if(x1 + r > img_width or y1 + r > img_height): return
            return (x1/img_width, y1/img_height, r/img_height)
        else:
            if random.random() > 0.5:
                w = size * img_width
                h = ratio * w
            else:
                h = size * img_height
                w = ratio * h
            if(x1 + w > img_width or y1 + h > img_height): return
            return (x1/img_width, y1/img_height, w/img_width, h/img_height)

    def generate_diagrams(self, quantity, directory='./generated/'):
        start = time.perf_counter()

        for i in range(quantity):
            W, H = 7168, 4567
            img = Image.new('RGB', (W, H), 'white')
            draw = ImageDraw.Draw(img)
            file_name = f'{directory}{i}.png'

            radial_symbols = {
                'heat_exchanger': random.randint(1, 2),
                'controller': random.randint(20, 30)
            }

            linear_symbols = {
                'valve': random.randint(5, 8),
                #'bell_valve': random.randint(4,6),
                'vessel': random.randint(1, 3)
            }

            bounding_boxes = []

            for key in radial_symbols:
                for _ in range(radial_symbols[key]):
                    bbox = None
                    for _ in range(20000):
                        cand = self._generate_bounding_box(key, W, H)
                        if cand and self._valid_box(
                            (cand[0], cand[1], 2 * cand[2], 2 * cand[2]), 
                            bounding_boxes):
                            bbox = cand
                            break

                    if not bbox:
                        continue

                    if key == 'heat_exchanger':
                        self.generate_heat_exchanger(img, draw, *bbox)
                    if key == 'controller':
                        self.generate_controller(img, draw, *bbox)

                    bounding_boxes.append((bbox[0], bbox[1], 2 * bbox[2], 2 * bbox[2]))

            for key in linear_symbols:
                for _ in range(linear_symbols[key]):
                    bbox = None
                    for _ in range(20000):
                        cand = self._generate_bounding_box(key, W, H)
                        if cand and self._valid_box(cand, bounding_boxes):
                            bbox = cand
                            break

                    if not bbox:
                        continue

                    if key == 'valve':
                        self.generate_valve(img, draw, *bbox)
                    if key == 'vessel':
                        self.generate_vessel(img, draw, *bbox)

                    bounding_boxes.append(bbox)

            img.save(file_name)

        print(f"Computation time: {time.perf_counter() - start}")

gen = DatasetGenerator('black')
gen.generate_diagrams(10)