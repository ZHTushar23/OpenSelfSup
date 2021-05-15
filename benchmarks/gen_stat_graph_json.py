import numpy as np
import glob, os
import matplotlib.pyplot as plt
import argparse
import json


def extract_info_from_log_file(logfile):
    # Extract the total number of epochs
    num_epochs = 200

    # Create empty array to store loss and acc values
    train_loss = np.zeros([num_epochs])
    train_acc = np.zeros_like(train_loss)
    top1_acc = np.zeros([num_epochs])
    val_loss = np.zeros([num_epochs])

    for epoch in range(num_epochs):
        tmp_loss = []
        tmp_acc = []
        for line in logfile:
            for name in line.keys():
                if name == 'epoch' and line[name] == epoch + 1:
                    if line["mode"] == "train":
                        tmp_loss.append(float(line['loss']))
                        tmp_acc.append(float(line['acc']))
                        # train_loss[epoch] = float(line['loss'])
                        # train_acc[epoch] = float(line['acc'])

                    elif line["mode"] == "val":
                        top1_acc[epoch] = float(line['head0_top1'])
                        val_loss[epoch] = float(line['val_loss'])

        train_loss[epoch] = np.array(tmp_loss).mean()
        train_acc[epoch] = np.array(tmp_acc).mean()

    return train_loss, train_acc, top1_acc, val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default=None, help='working directory missing')
    parser.add_argument('--pretrained', default=None, help='working directory missing')

    args = parser.parse_args()

    os.chdir(args.work_dir)

    for file in glob.glob("*.json"):
        logfile_name = file
    # Opening JSON file
    logfile = []
    for line in open(logfile_name, 'r'):
        logfile.append(json.loads(line))

    train_loss, train_acc, top1, val_loss = extract_info_from_log_file(logfile)

    plt.figure(1)
    plt.plot(train_loss, label='classifier training loss, min loss: ' + str(np.min(train_loss)))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.title(f'Pretrained: {args.pretrained}')
    plt.savefig('train_loss.png')

    plt.figure(2)
    plt.plot(train_acc, label='classifier training accuracy, max acc: ' + str(np.max(train_acc)))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title(f'Pretrained: {args.pretrained}')
    plt.savefig('train_acc.png')

    plt.figure(3)
    plt.plot(top1, label='classifier top1 score, max score: ' + str(np.max(top1)))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title(f'Pretrained: {args.pretrained}')
    plt.savefig('top1_acc.png')

    plt.figure(4)
    plt.plot(val_loss, label='classifier val_loss, min loss: ' + str(np.min(val_loss)))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.title(f'Pretrained: {args.pretrained}')
    plt.savefig('val_loss.png')

    # plt.show()
