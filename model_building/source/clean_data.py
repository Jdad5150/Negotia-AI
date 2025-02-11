from kaggle.api.kaggle_api_extended import KaggleApi
import os
import pandas as pd


def download_dataset():
    """
    This function downloads the LinkedIn Job Postings dataset from Kaggle and saves it to the data directory.
    """
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

    api = KaggleApi()
    api.authenticate()

    dataset = 'arshkon/linkedin-job-postings'
    save_path = os.path.join(os.getcwd(),'model_building', 'data')
    os.makedirs(save_path, exist_ok=True)

    print(f'Downloading dataset: {dataset} to {save_path}')
    api.dataset_download_files(dataset,path=save_path, unzip=True)
    print('Download complete.')


    
if __name__ == '__main__':

    # Make sure you are in the correct directory
    # you should be in 'model_building'
    print(os.getcwd())


    needs_data = input('Do you need to download the data? Press 1 for yes, 0 for no: ')
    
    if needs_data == '1':
        download_dataset()
    else:
        print('Data download skipped.')

    file_name = 'postings.csv'

    df = pd.read_csv('data/' + file_name)
    print(df.info())
