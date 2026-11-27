from trainer.train import Trainer
from argparse import ArgumentParser

def args_parser():
    parser = ArgumentParser()
    parser.add_argument('--training_dataset', type = str, default = 'barissglc/tarot', help = 'Dataset to train the tarot reader model')
    parser.add_argument('--contextual_dataset', type = str, default = 'barissglc/tarot', help = 'Dataset to add RAG context to the tarot reader', required=False)
    parser.add_argument('--base_model_name', type = str, default = 'meta-llama/Llama-3.2-3B-Instruct', help = 'Model being trained on the tarot data')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    trainer = Trainer(args.training_dataset, args.contextual_dataset)
    trainer.HubLogin()
    trainer.train(args.base_model_name)