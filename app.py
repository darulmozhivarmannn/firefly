import os
import base64
import json

from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

# Store this key in Render's environment variables.
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = os.environ.get(
    "CLAUDE_MODEL",
    "claude-sonnet-4-5-20250929"
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "MCQ Reader"
    })


@app.post("/answer")
def answer():
    image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({"error": "No image received"}), 400

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image too large"}), 413

    # The ESP32 firmware sends a JPEG as the raw HTTP request body.
    if not image_bytes.startswith(b"\xff\xd8"):
        return jsonify({"error": "Expected a JPEG image"}), 400

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
Read the multiple-choice question and all visible answer choices
in this image.

Return exactly one JSON object with these fields:
{
  "answer": "A",
  "explanation": "A short explanation."
}

The answer must be exactly A, B, C, or D.
If the question or choices are unreadable, do not guess.
Instead return:
{
  "answer": "?",
  "explanation": "Please retake the photo."
}

Keep the explanation short enough for a small OLED display.
Do not include Markdown fences or any text outside the JSON object.
"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        text_blocks = [
            block.text
            for block in response.content
            if block.type == "text"
        ]

        if not text_blocks:
            return jsonify({
                "answer": "?",
                "explanation": "No answer returned."
            }), 502

        raw_text = "\n".join(text_blocks).strip()

        # Be tolerant if the model unexpectedly wraps JSON in code fences.
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        result = json.loads(raw_text)

        answer_value = str(result.get("answer", "?")).strip().upper()
        if answer_value not in ("A", "B", "C", "D", "?"):
            answer_value = "?"

        explanation = str(result.get("explanation", ""))[:300]

        return jsonify({
            "answer": answer_value,
            "explanation": explanation
        })

    except json.JSONDecodeError:
        app.logger.exception("Claude returned invalid JSON")
        return jsonify({
            "answer": "?",
            "explanation": "Could not read AI response."
        }), 502

    except Exception:
        app.logger.exception("MCQ request failed")
        return jsonify({
            "answer": "?",
            "explanation": "Server or AI error."
        }), 502
