import os


def find_main_emotion(text):
    '''This function finds and returns the
    main emotion in the text from a .ann file'''

    for i in range(len(text)):
        if 'Main' in text[i]:
            return text[i-1].split()[1]

    return 'None'


def main():

    folderpath = 'adju_round_1_fixed/adju-1-2'
    output_file = 'trainset.txt'
    output_list = []

    trainset_info = []

    for group_number in sorted(os.listdir(folderpath)):
        for i in range(1, 51):
            file_info = []

            ann_file = (
                folderpath + '/' + group_number +
                '/' + '{:0>2}'.format(i) + '.ann'
                )
            with open(ann_file, 'r') as inp:
                text = inp.readlines()

            file_info.append(find_main_emotion(text))
            file_info.append(
                group_number + '/' + '{:0>2}'.format(i) + '.txt'
                )

            text_file = (
                folderpath + '/' + group_number +
                '/' + '{:0>2}'.format(i) + '.txt'
                )

            with open(text_file, 'r') as inp:
                file_info.append(inp.read())

            trainset_info.append(file_info)

    for file_info in trainset_info:
        file_info = ' '.join(file_info)
        file_info = file_info.replace('\n', ' ')
        output_list.append(file_info)

    with open(output_file, 'w') as outp:
        outp.write('\n'.join(output_list))


if __name__ == "__main__":
    main()
