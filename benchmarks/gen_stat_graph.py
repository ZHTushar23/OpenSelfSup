import numpy as np
import glob, os
import matplotlib.pyplot as plt
import argparse

def extract_info_from_log_file(logfile):
    #Extract the total number of epochs
    with open(logfile, 'r') as f:
        for line in f:
            if 'total_epochs' in line:
                idx = line.find("total_epochs")
                num_epochs = int(line[idx + 15:])

    # Create empty array to store loss and acc values

    balanced_acc  = []

    with open(logfile, 'r') as f:
        for line in f:
            if "balanced_acc" in line:
                balanced_acc.append(extract_acc_info(line))

    return np.array(balanced_acc)

def extract_acc_info(line):
    idx = line.find("balanced_acc")
    # print(idx)
    return float(line[idx+13:idx+20])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir',default=None,help='working directory missing')
    parser.add_argument('--pretrained', default=None, help='working directory missing')

    args = parser.parse_args()

    os.chdir(args.work_dir)

    for file in glob.glob("*.log"):
        if file.startswith("train"):
            logfile_name = file

    balanced_acc =  extract_info_from_log_file(logfile_name)

    plt.figure(1)
    plt.plot(balanced_acc, label='balanced accuracy, max acc: ' + str(np.max(balanced_acc)))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title(f'Pretrained: {args.pretrained}')
    plt.savefig('balanced_acc.png')
    # plt.show()
