import json
import re
import csv
import shutil
import os
import sys

main_path = os.getcwd()

def prepare_data_for_model(path):
    f = open(path, 'r')
    data = csv.DictReader(f)
    data_lines = []
    for row in data:
        phoneme = row['phenome']
        utterance_name = row['seg_id']
        speake_id = row['speaker_id']
        phoneme = re.sub("\[([0-9]+)\]", '', phoneme)
        phoneme = re.sub("\s+\|\s+", ' ', phoneme)
        data_lines.append([phoneme, utterance_name, speake_id])
    f.close()
    return data_lines
    

def save_files(train_data, test_data, data_path):
    for line in train_data:
        try:
            original = os.path.join(data_path, 'train_wav/{}.wav'.format(line[1]))
            target = os.path.join(main_path, 'dataset/persian_data/train_data/book-1/speaker-{0}/utterance-{1}.wav'.format(line[2], line[1]))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(original, target)
        except Exception as e:
            print(e)
            return False

        path = os.path.join(main_path, 'dataset/persian_data/train_data/book-1/speaker-{0}/utterance-{1}.txt'.format(line[2], line[1]))
        with open(path, 'w') as fp:
            fp.write(line[0])

    for line in test_data:
        try:
            original = os.path.join(data_path, 'test_wav/{}.wav'.format(line[1]))
            target = os.path.join(main_path, 'dataset/persian_data/test_data/book-1/speaker-{0}/utterance-{1}.wav'.format(line[2], line[1]))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(original, target)
        except Exception as e:
            print(e)
            return False

        path = os.path.join(main_path, 'dataset/persian_data/test_data/book-1/speaker-{0}/utterance-{1}.txt'.format(line[2], line[1])
        with open(path, 'w') as fp:
            fp.write(line[0])
    return True
    
def main(data_path):
    if os.path.isfile(os.path.join(data_path, 'train_info.csv')):
        train_data_path = os.path.join(data_path, 'train_info.csv')
    else:
        print('data_path is not correct!')
        return -1
    if os.path.isfile(os.path.join(data_path, 'test_info.csv')):
        test_data_path = os.path.join(data_path, 'test_info.csv')
    else:
        print('data_path is not correct!')
        return -1
    train_data = prepare_data_for_model(train_data_path)
    test_data = prepare_data_for_model(test_data_path)
    print('number of train data: ' + str(len(train_data)))
    print('number of test data: ' + str(len(test_data)))
    
    res = save_files(train_data, test_data, data_path)
    if res:
        print('Data is created.')

if __name__ == "__main__":
    main(sys.argv[1])