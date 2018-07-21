import os


def parse_dataset(data_dir):
    classes = ['pos', 'neg']

    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv9'):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)
    return train_data, train_labels, test_data, test_labels
