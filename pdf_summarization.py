# pip install transformers
# pip install PyPDF2
# pip install rouge-score
#ABSTRACTIVE SUMM
import PyPDF2
from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words
import os
import torch
from rouge_score import rouge_scorer
from pdfrw import PdfReader
import pandas as pd

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def extractPDFTitle(path, fileName):
    fullName = os.path.join(path, fileName)
    # Extract pdf title from pdf file
    Name = PdfReader(fullName).Info.Title
    # Remove surrounding brackets that some pdf titles have
    Name = Name.strip('()')
    return Name


# ## BART ###
# model_to_use='facebook/bart-large-cnn'# for T5 'google-t5/t5-base' # for BART: 'facebook/bart-large-cnn'
# token_to_use='facebook/bart-large-cnn'# for T5 'google-t5/t5-base' # for BART: 'facebook/bart-large-cnn'
# #3.Specify model, tokenizer and parameters
# min_in_length=10
# max_in_length=1024
# model=AutoModelForSeq2SeqLM.from_pretrained(model_to_use)
# tokenizer = AutoTokenizer.from_pretrained(model_to_use,model_max_length=max_in_length,model_min_length=min_in_length)
# extract_model = Summarizer()

## LSA ###
# summarizer = LsaSummarizer()
# summarizer.stop_words = get_stop_words("english")

##BERT##
# Load model directly
min_in_length=10
max_in_length=1024
tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
 
path='C:/Users/Stelios/Desktop/pdfs4'
for file_name in os.listdir(path):
      print(file_name)
      if file_name.lower().endswith('.pdf'):
          pdf_path = os.path.join(path, file_name)
          pdf_text = extract_text_from_pdf(pdf_path)
          file_without_extension = os.path.splitext(file_name)[0]
          file_with_txt = file_without_extension + ".txt"
          txt_path = os.path.join(path, file_with_txt )
          with open(txt_path, 'r') as file:
              file_abstract = file.read()
          input_txt=pdf_text   
          #summary = summarizer(input_txt)[0]['summary_text']
         # summary = summarizer(input_txt, min_length=30,do_sample=True)[0]['summary_text']

          #pdftitle= extractPDFTitle(path, file_name)
          #print(pdf_text[:1000])  # Print the first 1000 characters to verify correct reading
          #encoded_input = tokenizer(input_txt, padding='max_length', truncation=True, max_length=max_in_length, return_tensors='pt')

          ## BART and BERT ###
          encoded_input = tokenizer(input_txt, padding='max_length', truncation=True, return_tensors='pt')
          token_ids=encoded_input.input_ids
          attention_mask=encoded_input.attention_mask
          summarizer_ids=model.generate(token_ids,max_length=300,min_length=200)
          summary=tokenizer.decode(summarizer_ids[0])
          
          # ## LSA ###
          # parser = PlaintextParser.from_string(input_txt, Tokenizer("english"))
          # summary = summarizer(parser.document, sentences_count=4)  
          # full_sum="#".join(str(sentence) for sentence in summary) #For LSA
          # summary=full_sum

          # extsummary=extract_model(input_txt,num_sentences=20)
          print("\n\n-File Abstract:\n")
          print(file_abstract)
          print(f"\n-Extracted File Summary:\n {summary}")
          
          scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
          reference_sum=file_abstract
          results=scorer.score(reference_sum, summary)
          print("\n-Rouge1 Metrics:")
          print(results)
          rouge1_vals=list(results.values())[0]         
          rouge1_vals = [[item] for item in rouge1_vals]

          scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
          results=scorer.score(reference_sum, summary)
          print("\n-RougeL Metrics:")
          print(results)
          rouge2_vals=list(results.values())[0]         
          rouge2_vals = [[item] for item in rouge2_vals]
          
          file_path = 'rougebert4.xlsx' #FILE NAME
          df = pd.read_excel(file_path, sheet_name='Sheet1')
          all_vals=rouge1_vals+rouge2_vals
          rougedf = pd.DataFrame(all_vals)
          rougedf =rougedf.T
          updated_df = pd.concat([df, rougedf], ignore_index=True)
          with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
              updated_df.to_excel(writer, sheet_name='Sheet1', index=False)

          # rougedf =rougedf.T
          # rougedf.to_excel('output.xlsx', index=False,header=False)
         # print(f"Extractive File Summary:\n {extsummary}")
         









