from utils import load_pkl
import re
import json
def get_vocab_json():
    # apiseq_list = load_pkl('java_data/apiseq.pkl')
    # API_vocab_list = get_API_vocab_list(apiseq_list)
    # list_to_vocab_json(API_vocab_list, 'java_data/vocab.api.json')

    api2doc_dict = load_pkl('java_data/api2doc.pkl')
    APIdoc_vocab_list = get_APIdoc_vocab_list(api2doc_dict)
    list_to_vocab_json(APIdoc_vocab_list, 'java_data/vocab.apidoc.json')
    



def split_camel_case(word):
    return re.findall(r'[A-Z|a-z](?:[a-z]+|[A-Z]*[0-9]+)?', word)

def get_API_vocab_list(apiseq_list):
    API_vocab_list = []
    for api_seq in apiseq_list:
        for api in api_seq:
            apis = api.split(".")
            for api_item in apis:
                api_vocab_item = split_camel_case(api_item)
                for item in api_vocab_item:
                    if item not in API_vocab_list:
                        API_vocab_list.append(item)
    return API_vocab_list 

def get_APIdoc_vocab_list(api2doc_dict):
    APIdoc_vocab_list = []
    for api, doc in api2doc_dict.items():
        for item in doc:
            if item not in APIdoc_vocab_list:
                APIdoc_vocab_list.append(item)
    return APIdoc_vocab_list


def list_to_vocab_json(vocab_list, vocab_file):
    vocab_dict = {}
    vocab_dict["<pad>"] = 0
    vocab_dict["<s>"] = 1
    vocab_dict["</s>"] = 2
    vocab_dict["<unk>"] = 3
    for i in range(len(vocab_list)):
        vocab_dict[vocab_list[i]] = i+4

    
    
        
    with open(vocab_file, 'w') as f:
        json.dump(vocab_dict, f)

if __name__ == '__main__':
    #get_vocab_json()

    desc_dict = load_pkl('java_data/desc.pkl')
    print(1)