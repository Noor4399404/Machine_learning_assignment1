import argparse
import pprint
from sklearn.metrics import cohen_kappa_score


def main_emotions_collecter(folderpath):

    main_emotions = {}

    anno_folders = ('anno-1', 'anno-2')
    for anno in anno_folders:
        main_emotions[anno] = {}
        for i in range(1, 51):
            file = folderpath + '/' + anno + '/' + '30' + '/' + '{:0>2}'.format(i) + '.ann'
            with open(file) as inp:
                main_emotion = main_emotion_finder(inp.readlines())
                main_emotions[anno]['{:0>2}'.format(i)] = main_emotion

    return main_emotions


def main_emotion_finder(ann_text):

    for line_index in range(len(ann_text)):
        line = ann_text[line_index].split()
        if 'Main' in line:
            return ann_text[line_index - 1].split()[1]
    
    return 'Other'


def main():    

    parser = argparse.ArgumentParser(
        description="Retrieve all the main emotions for assigment 1B")
    parser.add_argument(
        "filepath",
        help="The path to the unzipped individual_round folder.")


    args = parser.parse_args()

    main_emotions = main_emotions_collecter(args.filepath)

    #pp = pprint.PrettyPrinter()
    #pp.pprint(main_emotions)

    cohen_kappa = cohen_kappa_score(
        list(main_emotions['anno-1'].values()), 
        list(main_emotions['anno-2'].values())
        )

    print(f'The cohen kappa score is: {cohen_kappa}')


if __name__ == "__main__":
    main() 