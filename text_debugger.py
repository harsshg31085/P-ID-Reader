import fitz
import pytesseract
from PIL import Image
import io
import json

def extract_text_blocks(pdf_path: str):
    doc = fitz.open(pdf_path)
    results = []

    for page_index, page in enumerate(doc):
        vector_text = page.get_text('dict')

        if vector_text:
            for block in vector_text['blocks']:
                if 'lines' not in block: continue
                for line in block['lines']:
                    for span in line['spans']:
                        x0, y0, x1, y1, = span['bbox']

                        results.append({
                            'page': page_index+1,
                            'x': x0,
                            'y': y0,
                            'width': x1-x0,
                            'height': y1-y0,
                            'text': span['text'],
                            'source': 'vector'
                        })
                
        else:
            pix = pix.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    x = data['left'][i]      
                    y = data['top'][i]      
                    w = data['width'][i]      
                    h = data['height'][i] 

                    pdf_w = page.rect.width
                    pdf_h = page.rect.height
                    scale_x = pdf_w / img.width
                    scale_y = pdf_h / img.height

                    results.append({
                        "page": page_index+1,
                        "x": x * scale_x,
                        "y": y * scale_y,
                        "width": w * scale_x,
                        "height": h * scale_y,
                        "text": data['text'][i],
                        "source": "OCR"
                    })
    return results

blocks = extract_text_blocks('input.pdf')
with open('e.json', 'w') as f:
    json.dump(blocks, f, indent=4)