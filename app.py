from flask import Flask, request, jsonify, render_template
from skills_extractor import get_skills_from_cv
from matcher import recommend_projects
import tempfile
import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/upload-cv", methods=["POST"])
def upload_cv():
    if "cv" not in request.files:
        return jsonify({"error": "No CV uploaded"}), 400

    file = request.files["cv"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        file.save(temp_pdf.name)
        skills_json = get_skills_from_cv(temp_pdf.name)

    return jsonify({"extracted_skills": skills_json})

@app.route("/recommend-projects", methods=["POST"])
def match_projects():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        skills = data.get("skills", [])
        interests = data.get("interests", "")
        experience = data.get("experience", "")

        if not skills:
            return jsonify({"error": "Missing skills"}), 400

        # Generate AI-powered recommendations
        recommendations = generate_ai_recommendations(skills, interests, experience)
        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_ai_recommendations(skills, interests, experience):
    """Use Cohere AI to generate personalized project recommendations"""
    prompt = f"""
    Based on the following profile:
    - Skills: {', '.join(skills)}
    - Interests: {interests}
    - Experience Level: {experience}

    Generate 3 personalized project recommendations that would help this person grow their skills.
    For each project, include:
    - A creative title
    - Detailed description
    - 3-5 key skills needed
    - Difficulty level matching their experience

    Return the response as a JSON array with this exact format:
    [
        {{
            "title": "",
            "description": "",
            "required_skills": [],
            "difficulty": ""
        }},
        ...
    ]
    """

    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=800,
        temperature=0.7,
        num_generations=1
    )

    try:
        # Extract JSON from the AI response
        generated_text = response.generations[0].text
        json_start = generated_text.find('[')
        json_end = generated_text.rfind(']') + 1
        json_str = generated_text[json_start:json_end]
        return eval(json_str)
    except Exception as e:
        print("Error parsing AI response:", str(e))
        # Fallback to default recommendations
        return get_fallback_recommendations(skills, interests, experience)

def get_fallback_recommendations(skills, interests, experience):
    """Fallback recommendations if AI fails"""
    return [
        {
            "title": f"AI {experience} Project: Intelligent Chatbot",
            "description": f"Build a {experience.lower()}-level AI chatbot that can understand and respond to user queries naturally.",
            "required_skills": ["Python", "NLP", "AI"],
            "difficulty": experience
        },
        {
            "title": f"Data Analysis {experience} Project",
            "description": f"Create a {experience.lower()}-level data analysis project using your skills in {', '.join(skills[:3])}.",
            "required_skills": skills[:3],
            "difficulty": experience
        },
        {
            "title": f"Skill Development {experience} Project",
            "description": f"A {experience.lower()}-level project combining your interests in {interests} with your technical skills.",
            "required_skills": skills[:3],
            "difficulty": experience
        }
    ]
if __name__ == "__main__":
    app.run(debug=True, port=5000)