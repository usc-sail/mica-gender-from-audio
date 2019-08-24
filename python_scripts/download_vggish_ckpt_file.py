'''
Author/year: Rajat Hebbar, 2019

Download checkpoint file for extracting VGGish embeddings
for gender ID prediction. This will be executed only if the
.ckpt file is not found in proj_dir/python_scripts/audio_scripts/

Input
    1) checkpoint : Path to .ckpt file 

Output
    1) checkpoint-file : Downloaded .ckpt file

Usage
    python download_vggish_ckpt_file.py checkpoint

Example
    python download_vggish_ckpt_file.py python_scripts/audioset_scripts/vggish_model.ckpt 
'''

import os, sys, requests

def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)

def main():
    if not os.path.isfile(checkpoint):
        print("Downloading VGGish model checkpoint file")
        download_file_from_google_drive('1c-wi6F_Fv0Z0TmJBpSrlTT0iCDmKF_NJ', checkpoint)

if __name__=='__main__':
    checkpoint=sys.argv[1]
    main()
