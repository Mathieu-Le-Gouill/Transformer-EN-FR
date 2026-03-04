import torch
from utils import translate_sentences
from transformers import AutoTokenizer
import os

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    eng_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    fr_tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    # Model saving path
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model
    model = torch.load(os.path.join(checkpoint_dir, "transformer.pt"), map_location=device)
    model.eval()

    # Test sentences
    english_sentences = [
        "It is often said that the early bird catches the worm, but sometimes patience is more valuable.",
        "Had they followed the instructions carefully, they might have avoided the costly mistake.",
        "The book, which was written in the 19th century, still resonates with readers today.",
        "The scientist, who had spent years studying climate change, finally published her groundbreaking research.",
        "Although it was raining heavily, she decided to go for a long walk in the park.",
        "She wondered whether she would ever have the courage to confront her fears.",
        "While waiting for the train, I noticed a group of children playing happily near the station.",
        "If I had known about the meeting earlier, I would have prepared a detailed presentation."
    ]

    french_translation = translate_sentences(model, eng_tokenizer, fr_tokenizer, english_sentences, device=device, max_len=50)

    for eng_sentence, fr_sentence in zip(english_sentences, french_translation):
        print(f"EN: {eng_sentence}")
        print(f"FR: {fr_sentence}\n")
