import os
import sys
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube, Playlist
from src.nlpyoutube.logger import logging
from src.nlpyoutube.exception import CustomException

class DataIngestion:
    def __init__(self):
        pass

    def get_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([entry['text'] for entry in transcript])
        except Exception as e:
            raise CustomException(e,sys)

    def fetch_video_details(self, video_url):
        try:
            yt = YouTube(video_url)
            return yt.video_id, yt.title, yt.thumbnail_url
        except Exception as e:
            raise CustomException(e,sys)

    
    def fetch_video_urls(self, url):
        if "playlist" in url:
            playlist = Playlist(url)
            return playlist.video_urls
        return [url]
