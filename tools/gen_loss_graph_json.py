import numpy as np
import glob, os
import matplotlib.pyplot as plt
import argparse
import json


def extract_loss_from_log_file(logfile):
    # Extract the total number of epochs
    num_epochs = 200

    # Create empty array to store loss and acc values
    loss     = np.zeros([num_epochs])
    val_loss = np.zeros([num_epochs])

    for epoch in range(num_epochs):
        tmp_loss = []
        for line in logfile:
            for name in line.keys():
                if name == 'epoch' and line[name]==epoch+1:
                    if 'loss' in line:
                        tmp_loss.append(float(line['loss']))
                    elif 'val_loss' in line:
                        val_loss[epoch] = float(line['val_loss'])

        loss[epoch] = np.array(tmp_loss).mean()
    return loss,val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default=None, help='working directory missing')
    parser.add_argument(
        '--pretrained', default=None, help='pretrained model file')
    parser.add_argument('--training_dataset', default='isic2017', help='working directory missing')
    args = parser.parse_args()

    os.chdir(args.work_dir)

    for file in glob.glob("*.json"):
        logfile_name = file

    # Opening JSON file
    logfile = []
    for line in open(logfile_name, 'r'):
        logfile.append(json.loads(line))

    loss, val_loss = extract_loss_from_log_file(logfile)
    pretext_task = "MOCO2"

    plt.figure(1)
    plt.plot(loss, label='backbone training loss, min loss: ' + str(np.min(loss)))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.title(f'Pretrained: {args.pretrained}\nBackbone Training: PT-{pretext_task}, DS: {args.training_dataset}')
    plt.savefig('loss.png')

    plt.figure(2)
    plt.plot(val_loss, label='backbone validation loss, min loss: ' + str(np.min(val_loss)))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.title(f'Pretrained: {args.pretrained}\nBackbone Training: PT-{pretext_task}, DS: {args.training_dataset} (Val)')
    plt.savefig('val_loss.png')
    # plt.show()

