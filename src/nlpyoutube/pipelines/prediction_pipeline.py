from src.nlpyoutube.components.data_ingestion import DataIngestion
from src.nlpyoutube.components.data_transformation import DataTransformation
from src.nlpyoutube.exception import CustomException
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import sys
class PredictionPipeline:
    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.data_transformation.load_tokenizer(tokenizer_path)
        self.model = load_model(model_path)

    def predict(self, video_url):
        try:
            videos = self.data_ingestion.fetch_video_urls(video_url)
            predictions = []

            for batch_start in range(0, min(10, len(videos)), 5):  # Process in batches of 5
                batch_videos = videos[batch_start:batch_start+5]
                batch_data = []

                for video in batch_videos:
                    video_id, title, thumbnail_url = self.data_ingestion.fetch_video_details(video)
                    transcript = self.data_ingestion.get_transcript(video_id)
                    preprocessed_text = self.data_transformation.preprocess(transcript)
                    keywords = self.data_transformation.extract_keywords(preprocessed_text)
                    preprocessed_sequence = self.data_transformation.preprocess_text(preprocessed_text)
                    
                    batch_data.append({
                        "title": title,
                        "thumbnail_url": thumbnail_url,
                        "keywords": keywords,
                        "preprocessed_sequence": preprocessed_sequence
                    })

                predictions_batch = self.model.predict(np.array([data['preprocessed_sequence'][0] for data in batch_data]))
                for i, prediction in enumerate(predictions_batch):
                    sentiment = np.argmax(prediction, axis=0)
                    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                    predictions.append({
                        "title": batch_data[i]['title'],
                        "thumbnail_url": batch_data[i]['thumbnail_url'],
                        "sentiment": sentiment_labels[sentiment],
                        "keywords": batch_data[i]['keywords']
                    })

            return predictions
        except Exception as e:
            raise CustomException(e,sys)


    def recommend_videos(self, video_url, video_choice):
        try:

            videos = self.data_ingestion.fetch_video_urls(video_url)
            videos_data = []

            for video in videos[:10]:

                video_id, title, thumbnail_url = self.data_ingestion.fetch_video_details(video)
                transcript = self.data_ingestion.get_transcript(video_id)
                preprocessed_text = self.data_transformation.preprocess(transcript)
                keywords = self.data_transformation.extract_keywords(preprocessed_text)
                videos_data.append({
                    "title": title,
                    "url": video,
                    "keywords": keywords,
                    "thumbnail_url": thumbnail_url
                })

            data = pd.DataFrame(videos_data)
            data['keywords_processed'] = data['keywords'].apply(lambda x: ' '.join(x))
            
            cosine_sim_matrix = self.data_transformation.calculate_cosine_similarity(data['keywords_processed'])
            
            video_titles = data['title'].tolist()
            video_index = video_titles.index(video_choice)

            sim_scores = list(enumerate(cosine_sim_matrix[video_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]  # Get top 5 similar videos
            video_indices = [i[0] for i in sim_scores]
            recommendations = data.iloc[video_indices][['title', 'url', 'keywords', 'thumbnail_url']]

            return {'recommendations': recommendations.to_dict('records')}
        except Exception as e:
            raise CustomException(e,sys)
