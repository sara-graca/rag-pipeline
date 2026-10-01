from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


PDF_PATH = Path("data/raw/Core Rules.pdf")
OUTPUT_DIR = Path("data/processed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"pdftext_workers": 1},
    )

    print("Converting...")
    rendered = converter(str(PDF_PATH))
    text, _, images = text_from_rendered(rendered)

    (OUTPUT_DIR / "parsed_rules.md").write_text(text, encoding="utf-8")

    for name, image in images.items():
        image.save(OUTPUT_DIR / name)

    print("Done.")


if __name__ == "__main__":
    main()