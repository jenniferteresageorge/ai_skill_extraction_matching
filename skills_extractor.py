import fitz  # PyMuPDF
import cohere
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_json_array(text):
    # Match the first JSON array in the string
    match = re.search(r'\[\s*{.*?}\s*]', text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No valid JSON array found in response.")

def categorize_skills_with_cohere(cv_text):
    system_message = "You are a professional CV analyzer that categorizes skills from resumes."
    user_prompt = f"""
Extract and categorize all skills from this CV into a JSON format like below:

[
  {{
    "Skill Category": "Soft Skills",
    "Skills": ["Communication", "Teamwork"]
  }},
  {{
    "Skill Category": "Technical Skills",
    "Skills": ["Python", "Java"]
  }}
]

CV TEXT:
\"\"\"
{cv_text}
\"\"\"
"""

    response = co.chat(
        model="command-r-plus",
        message=user_prompt,
        temperature=0.3,
        chat_history=[],
        prompt_truncation='AUTO',
    )

    generated = response.text.strip()
    print("COHERE OUTPUT:\n", generated)  # Optional: for debug

    try:
        cleaned = extract_json_array(generated)
        return json.loads(cleaned)
    except Exception as e:
        print("Error while parsing JSON:", str(e))
        raise

def get_skills_from_cv(pdf_path):
    cv_text = extract_text_from_pdf(pdf_path)
    return categorize_skills_with_cohere(cv_text)
