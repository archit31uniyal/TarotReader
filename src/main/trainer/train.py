from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import os
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer

class Trainer:
    def __init__(self, training_dataset_name, contextual_dataset_name):
        self.training_dataset_name: str = training_dataset_name
        self.contextual_dataset_name: str = contextual_dataset_name

    def HubLogin(self):
        with open(f'{os.getcwd()}/huggingface_access_token.txt', 'r') as fp:
            key = fp.read()
            login(key)

    def download_dataset(self, dataset_name, split = 'train'):
        dataset = load_dataset(dataset_name, split = split)
        print(f'Dataset: {dataset_name}, Split: {split} successfully downloaded')
        return dataset

    def load_tokenizer(self, model_name):
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f'Tokenizer successfully loaded for {model_name}')
        return base_tokenizer

    def train(self, student_model_name, teacher_model_name):
        dataset = self.download_dataset(self.training_dataset_name, "train")
        training_args = MiniLLMConfig(
            rkl_advantage=True,
            single_step_decomposition=False,
            gamma=False
        )
        model_trainer = MiniLLMTrainer(
            model = student_model_name,
            teacher_model = teacher_model_name,
            args = training_args,
            train_dataset = dataset
        )
