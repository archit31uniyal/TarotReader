from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import os

class Trainer:
    def __init__(self, training_dataset_name, contextual_dataset_name):
        self.training_dataset_name: str = training_dataset_name
        self.contextual_dataset_name: str = contextual_dataset_name

    def HubLogin(self):
        with open(f'{os.getcwd()}/huggingface_access_token.txt', 'r') as fp:
            key = fp.read()
            login(key)

    def download_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        print(f'Dataset {dataset_name} successfully downloaded')
        return dataset

    def load_model_and_tokenizer(self, model_name):
        base_model = AutoModel.from_pretrained(model_name)
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f'Model and tokenizer successfully loaded for {model_name}')
        return base_model, base_tokenizer

    def train(self, model_name):
        dataset = self.download_dataset(self.training_dataset_name)
        model, tokenizer = self.load_model_and_tokenizer(model_name)







