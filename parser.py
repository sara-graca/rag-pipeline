from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter
from marker.output import text_from_rendered

# 1. Initialize models (loads OCR, layout detection, and LaTeX models)
artifact_dict = create_model_dict()

# 2. Create the converter instance
converter = PdfConverter(
    artifact_dict=artifact_dict,
)

# 3. Render the PDF document
rendered = converter(r"data\raw\Core Rules.pdf")

# 4. Extract Markdown text, metadata, and images
full_text, metadata, images = text_from_rendered(rendered)

# 5. Save output text and extracted images
with open(r"data\processed\parsed_rules.md", "w", encoding="utf-8") as f:
    f.write(full_text)

# Save images extracted from the PDF
for filename, image in images.items():
    image.save(f"./{filename}")