import argparse
import deepdoctection as dd
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import json
import random
import pytesseract

analyzer = dd.get_dd_analyzer()  

df = analyzer.analyze(path = "bosch-glm-20-manual.pdf")
df.reset_state()
doc = iter(df)
page = next(doc) 

image = page.viz()
image_path = "marked_image.png"
report_path = "index.html"
text_content = page.text
list_text = [text for text in text_content if text != "\n" and text != " " and text != "[]"]
def analyze_pdf(pdf_path, text_content, ocr=False, lang='eng'):
    if ocr:
        text_content += "\n" + perform_ocr(page.image_path, lang)

    return image_path, text_content

def perform_ocr(image_path, lang='eng'):
    image = plt.imread(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def save_image(page, image_path):
    image = page.viz()
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(image_path, bbox_inches='tight', dpi=115)

def matplotlib_animation(image_path):
    plt.figure(figsize=(8, 6))
    plt.imread(image_path)
    plt.plot([1, 2, 3, 4], [2, 4, 9, 16], "ro")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Sample Graph")
    plt.savefig("graph.png", bbox_inches='tight', dpi=115)

def json_data(image_path):
    json_data = {
        "blocks": [
            {
                "coordinates": random.sample([[100, 100], [200, 200], [300, 300], [100, 100]], 4),
                "text": "This is a block of text."
            },
            {
                "coordinates": random.sample([[400, 400], [500, 500], [600, 600], [400, 400]], 4),
                "text": "This is another block of text."
            }
        ],
        "images": [image_path]
    }

    with open("index.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4)


    plt.savefig("graph.png", bbox_inches='tight', dpi=115)

def generate_html_report(image_path, text_content):
    html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <!-- Подключение пользовательских стилей -->
            <style>
                .full-width-text {
                    "width: 100%;"
                    "padding: 20px;"
                }

                /* Стиль для содержания */
                .table-of-contents {
                    "background: #f5f5f5;"
                    "padding: 20px;"
                    "border: 1px solid #ddd;"
                }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                <a class="navbar-brand" href="#">Ваш документ</a>
            </nav>
            <div class="container mt-4">
                <div class="row">
                    <div class="col-lg-8 col-md-12">
                        <h1>Глава 1</h1>
                        <div class="full-width-text">
                            <p>{text_content}</p>                            
                        </div>

                        <h2>Изображение</h2>
                        <div class="full-width-text">
                            <img src='marked_image.png' alt='Marked image' />
                        </div>

                        <h2>Таблица</h2>
                        <div class="full-width-text">
                            <p>Это таблица</p>
                            <table>
                                <tr>
                                    <h5><td>{text_content}</td></h5>
                                </tr>
                            </table>
                        </div>

                        <h2>Maтplotlib image</h2>
                        <div class="full-width-text">
                            <img src='graph.png' alt='Graph' />

                        </div>
                    </div>

                    <div class="col-lg-4 col-md-12">
                        <div class="table-of-contents">
                            <h3>Содержание</h3>
                            <ul>
                                <li><a href="#section-1">Глава 1</a></li>
                                <li><a href="#subsection-1">Изображение</a></li>
                                <li><a href="#subsection-2">Таблица</a></li>
                                <li><a href="#subsection-2">Matplotlib image</a></li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            
            
        </body>
        </html>
        """

    with open(report_path, "w") as html_file:
        html_file.write(html)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--ocr', action='store_true', help='Perform OCR')
    parser.add_argument('--lang', default='eng', help='Language for OCR (default: "eng")')
    parser.add_argument('--output_path', help='Path to save the HTML report')

    args = parser.parse_args()

    image_path, text_content = analyze_pdf(args.pdf_path, args.ocr, args.lang)

    save_image(page, image_path)
    json_data(image_path)
    generate_html_report(image_path, text_content)

    if args.output_path:
        with open(args.output_path, "w") as report_file:
            report_file.write(generate_html_report(image_path, text_content))
    else:
        HTML(generate_html_report(image_path, text_content))
