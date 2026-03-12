import pandas as pd
import csv
from model import InputFunction

# Helper
def getTokens():
    with open("./TokenDataset.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row['Tokens'] for row in reader]

class GeminiDescriptions(InputFunction):
    def __init__(self):
        super().__init__("Gemini Descriptions")
    
    def run(self) -> list[str]:
        with open("./TokenDataset.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [f"{row['Tokens']}: {row['Description (Gemini)']}" for row in reader]

class SynonymDescriptions(InputFunction):
    def __init__(self):
        super().__init__("Synonym Descriptions")
    
    def run(self) -> list[str]:
        with open("./TokenDataset.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [f"{row['Tokens']}: {row['Description Syn']}" for row in reader]
        
class ChatGPTDescriptions(InputFunction):
    def __init__(self):
        super().__init__("ChatGPT Descriptions")
    
    def run(self) -> list[str]:
        with open("./TokenDataset.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [f"{row['Tokens']}: {row['Description (ChatGPT)']}" for row in reader]
            