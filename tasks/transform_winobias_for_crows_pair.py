from pathlib import Path
import csv


DATASET_PATH = Path("data/winobias")

def transform_dataset():
    file_name_template = "{0}_stereotyped_type{1}.txt.test"

    sentence_types = {
        'ground': 1,
        'knowledge': 2,
    }

    for name, t_num in sentence_types.items():

        with open(DATASET_PATH / file_name_template.format("anti", t_num), 'r') as f:
            anti_dataset = f.readlines()

        with open(DATASET_PATH / file_name_template.format("pro", t_num), 'r') as f:
            pro_dataset = f.readlines()

        anti_dataset = [anti_sent.strip().replace("[", "").replace("]", "") for anti_sent in anti_dataset]
        pro_dataset = [pro_sent.strip().replace("[", "").replace("]", "") for pro_sent in pro_dataset]


        with open(f'data/winobias_{name}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['', 'sent_more', 'sent_less', 'stereo_antistereo'])

            for i, (anti_sent, pro_sent) in enumerate(zip(anti_dataset, pro_dataset)):
                writer.writerow([str(i), anti_sent, pro_sent, 'stereo'])


if __name__ == '__main__':
    transform_dataset()

