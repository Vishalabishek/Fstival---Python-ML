from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

def load_and_prepare_data(file_path, skill_columns):
    job_data = pd.read_csv(file_path)
    for column in skill_columns:
        if column in job_data.columns:
            job_data[column] = job_data[column].replace({'Yes': 1, 'No': 0})
            job_data[column] = pd.to_numeric(job_data[column], errors='coerce').fillna(0).astype(int)
    return job_data

def recommend_jobs(user_skills, job_data, skill_columns, top_n=5):
    user_skill_vector = np.zeros(len(skill_columns))
    for skill in user_skills:
        if skill in skill_columns:
            user_skill_vector[skill_columns.index(skill)] = 1

    job_skill_vectors = job_data[skill_columns].values
    similarity_scores = cosine_similarity([user_skill_vector], job_skill_vectors).flatten()

    job_data['Similarity'] = similarity_scores
    recommended_jobs = job_data.sort_values(by='Similarity', ascending=False).head(top_n)
    return recommended_jobs[['Job Title', 'Similarity']].to_dict(orient='records')

@app.route('/')
def home():
    return "Job Recommendation API is running!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_skills = data.get('skills', [])
    file_path = 'output-removed.csv' 

    skill_columns = ['Python', 'Java', 'C++', 'SQL', 'HTML', 'CSS', 'JavaScript', 'React', 
                     'Git', 'Agile', 'Machine Learning', 'Operating Systems', 'Version Control', 
                     'Cloud Platforms', 'Containerization', 'Data Structures & Algorithms', 
                     'API Development', 'Microservices Architecture', 'Cybersecurity', 'Big Data', 
                     'CI/CD Pipelines']

    job_data = load_and_prepare_data(file_path, skill_columns)
    recommended_jobs = recommend_jobs(user_skills, job_data, skill_columns, top_n=20)

    return jsonify(recommended_jobs)

if __name__ == '__main__':
    app.run(debug=True)