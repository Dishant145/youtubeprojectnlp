import os
from src.nlpyoutube.components.model_trainer import ModelTrainer
from src.nlpyoutube.utils import download_nltk_resources
import sys
def run_training_pipeline():
    download_nltk_resources()
    data_path = 'Artifact/transcripts.csv'
    tokenizer_path = 'tokenizer.pkl'
    model_path = 'model.h5'
    
    trainer = ModelTrainer()
    trainer.train_model(data_path, tokenizer_path, model_path)
