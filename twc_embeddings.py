import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import argparse
import json
import pdb
import torch
import torch.nn.functional as F

def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]


class HFModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.debug = False
        print("In HF Constructor")


    def init_model(self,model_name = None):
        # Get our models - The package will take care of downloading the models automatically
        # For best performance: Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit
        #print("Init model",model_name)
        if (model_name is None):
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_embeddings(self,input_file_name,input_data,is_file):
        #print("Computing embeddings for:", input_data[:20])
        model = self.model
        tokenizer = self.tokenizer

        texts = read_text(input_data) if is_file == True else input_data

        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return texts,sentence_embeddings

    def output_results(self,output_file,texts,embeddings,main_index = 0):
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        cosine_dict = {}
        #print("Total sentences",len(texts))
        for i in range(len(texts)):
                cosine_dict[texts[i]] = 1 - cosine(embeddings[main_index], embeddings[i])

        #print("Input sentence:",texts[main_index])
        sorted_dict = dict(sorted(cosine_dict.items(), key=lambda item: item[1],reverse = True))
        if (self.debug):
            for key in sorted_dict:
                print("Cosine similarity with  \"%s\" is: %.3f" % (key, sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict,indent=0))
        return sorted_dict



if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='SGPT model for sentence embeddings ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        parser.add_argument('-model', action="store", dest="model",default="sentence-transformers/all-MiniLM-L6-v2",help="model name")

        results = parser.parse_args()
        obj = HFModel()
        obj.init_model(results.model)
        texts, embeddings = obj.compute_embeddings(results.input,results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)
