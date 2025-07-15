import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
from chatbot_base import ChatbotBase
import re
import random as random
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas
import random as random
from datetime import datetime
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
import warnings 
warnings.filterwarnings("ignore")

class InstructLabourPolicyBot(ChatbotBase):
    def __init__(self, name="LabourPolicyBot", device='cpu'):
        ChatbotBase.__init__(self,name)
        self.device = device

        # Hyperparameters for text generation
        self.max_tokens = 200
        self.temperature = 0.2
        self.top_p = 0.99
        self.min_p = 0.6

        # Load the trained intent recognition model
        self.intent_model = SetFitModel.from_pretrained("ckpt/", local_files_only=True)

        #Loading datasets
        self.manifesto_df = pandas.read_csv('manifesto_info.csv')
        self.manifesto_topics = {row["Label"]: row["Answer"] for _, row in self.manifesto_df.iterrows()}
                
        # Checkpoint for LLM
        checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

        # Clean and process input text
    def process_input(self, user_input):
        # Remove leading and trailing whitespace characters
        processed_input = re.sub(r"[^\x00-\x7F]", "", user_input).strip().lower()
        return processed_input
    
        # Respond to user farewell
    def farewell(self, user_input):
        if user_input.lower() in ["goodbye", "bye", "exit", "quit"]:
            responses = [
                "Goodbye!",
                "Have a nice day!",
                "See you later!",
                "Take care, bye!"
            ]
            print(random.choice(responses))
            self.conversation_is_active = False  #Stop the conversation loop
            return True  #Indicate the conversation has ended
        return False
    
    #Classify the intent topic from training
    def classify_intent(self, processed_input):
        prediction = self.intent_model.predict([processed_input])
        return int(prediction[0])
    
    #Retrieve context using labels in the dataset, print response if intent label cannot be found
    def get_context(self, intent_label):
        return self.manifesto_topics.get(intent_label, "Sorry, I couldn't find an answer for you. Could you try rewording your question and I'll try again?")
    
    #System prompts
    def prepare_prompt(self, user_input, context):
        system_prompt = {
            "role": "system",
            "content":"You are an expert Labour Party policy bot. Answer the user's query factually based only on the context provided. If the context is insufficient, indicate what is missing, DO NOT ask questions, DO NOT respond to youself, only create responses under 200 characters."
            f"Context: {context}\n"
        }
        return f"{system_prompt}\nUser: {user_input}\nLabourPolicyBot:"

    #Respond using the LLM
    def respond_with_llm(self, user_input, context):
        prompt = self.prepare_prompt(user_input, context)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        output_text = self.tokenizer.decode(output[0][len(input_tokens[0]):], skip_special_tokens=True).strip()
        return self.clean_response(output_text)
    
    #Regex to clean the generated response to reduce risk of chatbot asking itself questions or mimicking user behaviour
    def clean_response(self, output_text):
        split_response = re.split(r'(User: | " )', output_text)
        clean_response = split_response[0].strip()
        return clean_response
    
    #Generate the response
    def generate_response(self, user_input):
        processed_input = self.process_input(user_input)
        intent_label = self.classify_intent(user_input)
        context = self.get_context(intent_label)
        return self.respond_with_llm(user_input, context)
    
    # Greeting function
    def greeting(self):
        print(f"\n"*100,f"Hello I am LabourPolicyBot, I'm here to help you explore and understand the Labour Party's policies and priorities for 2024. Simply type your question about a specific policy area or topic like healthcare, education, the economy, or climate change and I'll do my best to provide you with detailed answers based on Labour's official publications.  \n")

if __name__ == "__main__":
    llm_chatbot = InstructLabourPolicyBot()
    llm_chatbot.greeting()

    #Initialise first user input
    response = input("You: ")

    while llm_chatbot.conversation_is_active:
        # Check for farewell phrases before processing further
        if llm_chatbot.farewell(response):
            break  # Exit the loop if a farewell is detected

        # Process the user input and generate a response
        bot_response = llm_chatbot.generate_response(response)
        print(f"\nLabourPolicyBot: {bot_response}")

        # Get the next user input
        response = input("\nYou: ")
