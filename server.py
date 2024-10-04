from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import logging
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

app = Flask(__name__)
CORS(app, resources={r"/summarize": {"origins": "*"}})  # This will enable CORS for the /summarize route

client = MongoClient('mongodb://localhost:27017/')
db = client['url_summarizer']
collection = db['urls']

summarizer = pipeline("summarization")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/summarize', methods=['POST'])
def summarize():
    app.logger.debug("Received request to /summarize")
    data = request.json
    app.logger.debug(f"Request JSON: {data}")
    url = data.get('url')

    if not url:
        app.logger.error("No URL provided")
        return jsonify({"error": "No URL provided"}), 400
      
    # Store URL in MongoDB
    result=collection.insert_one({"url": url})
    app.logger.debug(f"Stored URL in MongoDB: {url}")

    # Retrieve and extract text from URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    text = ' '.join(paragraphs)
    print(text)
    print(len(text))
    #Perform summarization with BERTmax_length=200, min_length=30,
    
   # summary = summarizer(text[:5120],max_length=130, min_length=30,do_sample=False)[0]['summary_text']
    #collection.update_one({"_id":result.inserted_id },{"$set": {"summary": summary}})
    #app.logger.debug(f"Stored summary in MongoDB: {summary}")
    #print(summary)
    #return jsonify({"summary": summary})
    
    # Perform summarization with BaRT and T-5
    model_to_use='facebook/bart-large-cnn'# for T5 'google-t5/t5-base' # for BART: 'facebook/bart-large-cnn'
    token_to_use='facebook/bart-large-cnn'# for T5 'google-t5/t5-base' # for BART: 'facebook/bart-large-cnn'
    min_in_length=10
    max_in_length=1024
    model=AutoModelForSeq2SeqLM.from_pretrained(model_to_use)
    tokenizer = AutoTokenizer.from_pretrained(model_to_use,model_max_length=max_in_length,model_min_length=min_in_length)
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_in_length, return_tensors='pt')
    token_ids=encoded_input.input_ids
    attention_mask=encoded_input.attention_mask
    summarizer_ids=model.generate(token_ids,max_length=250,min_length=150)
    summary=tokenizer.decode(summarizer_ids[0])
    summary = summary.replace('</s><s>', ' ')
    summary = summary.replace('</s>', ' ')
    summary = summary.replace('<pad><extra_id_0>', ' ')
    summary = summary.replace('  Chrome. If you are unable to, please upgrade to a newer version of Firefox.', ' ')
    collection.update_one({"_id":result.inserted_id },{"$set": {"summary": summary}}) 
    app.logger.debug(f"Stored summary in MongoDB: {summary}")
    print(summary)
    return jsonify({"summary": summary})
    

    # # Load the Pegasus tokenizer
    # tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum',truncation=True)
    # # Load the Pegasus model for conditional generation
    # model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
    # # Tokenize input text, max_length=1024
    # inputs = tokenizer([text], return_tensors='pt', truncation=True)
    # summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # summary = summary.replace('</s><s>', ' ')
    # summary = summary.replace('</s>', ' ')
    # summary = summary.replace('<pad><extra_id_0>', ' ')
    # summary = summary.replace('  Chrome. If you are unable to, please upgrade to a newer version of Firefox.', ' ')
    # collection.update_one({"_id":result.inserted_id },{"$set": {"summary": summary}}) 
    # app.logger.debug(f"Stored summary in MongoDB: {summary}")
    # print(summary)
    # return jsonify({"summary": summary})
    

    # # # Perform extractive summarization with LSA
    #parser = PlaintextParser.from_string(text, Tokenizer("english"))
    #summarizer.stop_words = get_stop_words("english")
    #summary = summarizer(parser.document, sentences_count=4)  
    #print("\n-Extractive Summary:\n")
    #summary="#".join(str(sentence) for sentence in summary)
    #print(summary)
    #collection.update_one({"_id":result.inserted_id },{"$set": {"summary": summary}})   
    #app.logger.debug(f"Stored summary in MongoDB: {summary}")
    #print(summary)
    #return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)

