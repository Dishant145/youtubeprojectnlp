from flask import Flask, render_template, request
from src.nlpyoutube.pipelines.prediction_pipeline import PredictionPipeline
import pandas as pd

app = Flask(__name__)

pipeline = PredictionPipeline('model3.h5', 'tokenizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/keyword_search', methods=['GET', 'POST'])
def keyword_search():
    if request.method == 'POST':
        video_or_playlist_url = request.form['url']
        user_keywords = request.form['keywords']
        
        # Fetch predictions
        result = pipeline.predict(video_or_playlist_url)
        result_df = pd.DataFrame(result)
        
        # Prepare user keywords set for fast lookup
        user_keywords_set = set(map(str.strip, map(str.lower, user_keywords.split(','))))
        
        # Vectorized keyword matching
        matching_videos = result_df[result_df['keywords'].apply(lambda keywords: any(kw in user_keywords_set for kw in keywords))]
        
        return render_template('results.html', videos=matching_videos.to_dict('records'))
    
    return render_template('keyword_search.html')

@app.route('/video_recommendations', methods=['GET', 'POST'])
def video_recommendations():
    if request.method == 'POST':
        video_or_playlist_url = request.form['url']
        
        # Fetch predictions
        data = pipeline.predict(video_or_playlist_url)
        data_df = pd.DataFrame(data)
        
        video_titles = data_df['title'].tolist()
        video_choice = request.form.get('video_choice')
        
        if video_choice:
            video_index = video_titles.index(video_choice)
            
            # Get video recommendations
            result = pipeline.recommend_videos(video_or_playlist_url, video_index)
            return render_template('recommendations.html', recommendations=result['recommendations'], video_choice=video_choice)
        else:
            return render_template('video_recommendations.html', videos=video_titles)
    
    return render_template('video_recommendations.html', videos=[], video_url='')

if __name__ == '__main__':
    app.run(debug=True)
