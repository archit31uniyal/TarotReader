from huggingface_hub import login

def HubLogin():
    with open('/Users/archit/Documents/UpsellRecommender/huggingface_access_token.txt', 'r') as fp:
        key = fp.read(fp)
        login(key)