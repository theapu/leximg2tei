import os
import random
import io
import logging
from functools import wraps
from flask import Flask, request, Response, stream_with_context, jsonify
from google import genai
from google.genai import types
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Authentication Decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        service_key = os.environ.get("SERVICE_API_KEY")
        if not service_key:
            return jsonify({"error": "Server configuration error: SERVICE_API_KEY not set"}), 500

        request_key = request.headers.get("X-API-Key")
        if request_key and request_key == service_key:
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized: Invalid or missing API Key"}), 401
    return decorated_function

# Enhanced System Instruction for Strictness
SYSTEM_INSTRUCTION = """
You are the Chief Digitization Officer and TEI (Text Encoding Initiative) Architect specializing in historical Malayalam Lexicons and Manuscripts.
Your mission is to produce high-fidelity, archival-grade TEI P5 XML from scanned images.

### CRITICAL OBJECTIVE
Capture EVERY visible character on the page with 100% transcriptive accuracy.
**CRITICAL:** You must distinguish between **Dictionary Entries** and **Standard Prose/Tables**. Do not force standard paragraphs, tables, or lists into a dictionary structure.

### CONTENT TYPE RECOGNITION (Apply Logic Before Tagging)
Before tagging, analyze the visual layout of the page section:
1.  **Is it a Dictionary Entry?** (Headword, definition, grammar info) -> Use <entry> structure.
2.  **Is it Prose?** (Preface, introduction, footnotes, long descriptions) -> Use <p> and <note> structures.
3.  **Is it a Table?** (Grid layout, rows, columns of data) -> Use <table> structure.

### SCANNING & EXTRACTION RULES
1.  **Visual Hierarchy**: Maintain the reading order. Capture headers, page numbers, and marginalia first.
2.  **Columnar Logic**: Use <cb n="1"/> and <cb n="2"/> to mark the start of columns.
3.  **Malayalam Fidelity**: Transcribe Malayalam text exactly. Preserve archaic ligatures and complex conjuncts.
4.  **Phonetic Transcription**: Transcribe Latin phonetic text with complex diacritics (ā, ī, ū, ṛ, ḷ, ṉ, ṇ, ñ, ś, ṣ, ṯ, etc.) exactly. Do not normalize.

### HANDLING NON-LEXICAL CONTENT (Prose & Tables)
* **Prose/Paragraphs**: If the text is a preface, introduction, or narrative, strictly use <p>. Do not look for <orth> or <sense> inside standard sentences.
* **Tables**: If the image contains a grid or tabular data:
    * Use <table> to wrap the content.
    * Use <row> for horizontal lines.
    * Use <cell> for individual data points.
* **Lists**: Use <list> and <item> for non-dictionary vertical lists.
* **Page Markers**:
    * <pb/> for page breaks.
    * <fw type="pageNum"> for page numbers.
    * <fw type="header"> for running headers.

### HANDLING DICTIONARY ENTRIES (Only for Lexical Data)
Use this structure **ONLY** when distinct headwords and definitions are visible:
* <entry>: Wrapper for the word.
* <form>: Groups headword info. Contains <orth> (headword) and <pron> (phonetics).
* <gramGrp>: Contains <pos> (part of speech), <gen>, <number>.
* <sense>: Meaning units, numbered using @n.
* <def>: The literal definition string.
* <cit type="example">: Wrapper for usage examples. MUST be used instead of <eg>.
    * Inside <cit>, use <quote> for the example text.
    * Inside <cit>, use <bibl> for the source/author of the quote.
* <etym>: Etymology.

### ANTI-HALLUCINATION PROTOCOLS
1.  **Do NOT generate <entry> tags for introductory text.** If the page is an "Introduction," simply use <head> and <p>.
2.  **Do NOT invent structure.** If a section is just a list of names or numbers, use a <table> or <list>, not a dictionary entry.
3.  If the page contains **mixed content** (e.g., a paragraph followed by a table), transcribe them sequentially using the appropriate distinct tags.

### OUTPUT SPECIFICATIONS
1.  **Format**: Return valid, raw XML only. No Markdown code blocks (no triple backticks).
2.  **Dynamic Metadata**: You must identify the visible page number from the image and replace [INSERT PAGE NUMBER] in the header below.
3.  **Strict Skeleton**: You must strictly follow this root structure (note the xmlns attribute):

<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Malayalam Lexicon Digitization - Page [INSERT PAGE NUMBER]</title>
      </titleStmt>
      <publicationStmt>
        <p>Digital edition based on lexicon scans.</p>
      </publicationStmt>
      <sourceDesc>
        <bibl>Malayalam Lexicon, Page [INSERT PAGE NUMBER]</bibl>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
    </body>
  </text>
</TEI>
"""

def get_api_key():
    raw_keys = os.environ.get("API_KEYS") or os.environ.get("API_KEY")
    if not raw_keys:
        return None
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    return random.choice(keys)

@app.route("/convert", methods=["POST"])
@require_api_key
def convert_to_tei():
    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "Configuration Error: API_KEY not set."}), 500

    client = genai.Client(api_key=api_key)

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Default to 1.5-pro for better accuracy (less hallucination), even if slower.
    # Flash models (2.0-flash) are faster but prone to hallucinations on dense text.
    model_name = request.args.get("model") or os.environ.get("DEFAULT_MODEL", "gemini-1.5-pro")

    # Safety Settings: BLOCK_NONE is critical for historical text.
    # Filters often mistake archaic religious or cultural terms for policy violations,
    # leading to empty responses or hallucinations.
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        )
    ]

    def generate():
        try:
            # FIX: Use generate_content_stream method instead of stream=True parameter
            response = client.models.generate_content_stream(
                model=model_name,
                contents=[
                    "Transcribe this page into valid TEI XML strictly following the system instructions. Do not use Markdown.",
                    img
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1, # 0.0 forces the model to select the most probable token (less creative/hallucinatory)
                    safety_settings=safety_settings
                )
            )

            for chunk in response:
                # Basic filter to strip markdown if the model ignores the prompt
                text = chunk.text
                if text:
                    # Remove common markdown artifacts if they appear in chunks
                    text = text.replace("```xml", "").replace("```", "")
                    yield text

        except Exception as e:
            logger.error(f"GenAI Stream Error: {e}")
            yield f"\n<!-- Error during processing: {str(e)} -->"

    # Return a streamed response
    return Response(
        stream_with_context(generate()), 
        mimetype="application/xml",
        headers={"Content-Disposition": f"attachment; filename={file.filename}.xml"}
    )

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
