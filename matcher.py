from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def recommend_projects(user_data, project_data):
    """
    Enhanced recommendation system that combines:
    - AI-generated personalized projects
    - Semantic similarity matching
    """
    # First get AI-generated recommendations
    ai_recommendations = generate_ai_recommendations(
        user_data["skills"],
        user_data["interests"],
        user_data["experience"]
    )
    
    # Then calculate similarity with existing projects
    combined_projects = ai_recommendations + project_data
    
    # Create profile strings for similarity matching
    user_profile = create_user_profile(user_data)
    project_profiles = [create_project_profile(p) for p in combined_projects]
    
    # Calculate similarity
    vectorizer = TfidfVectorizer().fit([user_profile] + project_profiles)
    user_vector = vectorizer.transform([user_profile])
    project_vectors = vectorizer.transform(project_profiles)

    similarities = cosine_similarity(user_vector, project_vectors)[0]
    
    # Sort and return top 3 unique recommendations
    scored_projects = sorted(
        zip(combined_projects, similarities),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Remove duplicates and return top 3
    seen_titles = set()
    unique_recommendations = []
    for project, score in scored_projects:
        if project["title"] not in seen_titles:
            seen_titles.add(project["title"])
            unique_recommendations.append(project)
        if len(unique_recommendations) >= 3:
            break
    
    return unique_recommendations

def create_user_profile(user_data):
    """Create a comprehensive user profile text for similarity matching"""
    return (
        f"Skills: {', '.join(user_data['skills'])}. "
        f"Interests: {user_data['interests']}. "
        f"Experience: {user_data['experience']}"
    )

def create_project_profile(project):
    """Create a comprehensive project profile text for similarity matching"""
    return (
        f"{project['title']}. {project['description']}. "
        f"Skills needed: {', '.join(project['required_skills'])}. "
        f"Difficulty: {project.get('difficulty', '')}"
    )

def generate_ai_recommendations(skills, interests, experience):
    """Generate AI-powered project recommendations"""
    prompt = f"""
    Create 3 personalized project recommendations for someone with:
    - Skills: {', '.join(skills)}
    - Interests: {interests}
    - Experience: {experience}
    
    For each project include:
    - Creative title
    - Detailed description
    - Required skills (3-5)
    - Difficulty level
    
    Return as JSON array with format:
    [{{"title": "", "description": "", "required_skills": [], "difficulty": ""}}]
    """
    
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=800,
        temperature=0.7
    )
    
    try:
        # Extract JSON from response
        json_str = response.generations[0].text
        json_start = json_str.find('[')
        json_end = json_str.rfind(']') + 1
        return eval(json_str[json_start:json_end])
    except:
        return []