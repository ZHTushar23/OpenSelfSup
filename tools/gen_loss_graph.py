import numpy as np
import glob, os
import matplotlib.pyplot as plt
import argparse

def extract_task_name(logfile):
    #Extract the total number of epochs
    i=0 #loop break indicator
    with open(logfile, 'r') as f:
        for line in f:
            if 'model = dict(' in line:
                i=1
            elif i==1 and 'type' in line:
                tmp = (line.split("'"))
                pretext_task  = str(tmp[1])
                i=0
                break

    return pretext_task

def extract_loss_from_log_file(logfile):
    #Extract the total number of epochs
    with open(logfile, 'r') as f:
        for line in f:
            if 'total_epochs' in line:
                idx = line.find("total_epochs")
                num_epochs = int(line[idx + 15:])

    # Create empty array to store loss and acc values
    loss = np.zeros([num_epochs])
    for epoch in range(num_epochs):
        tmp_loss = []
        keyword1 = "Epoch ["+str(epoch+1)+"]"

        with open(logfile, 'r') as f:
            for line in f:
                if keyword1 in line:
                    tmp_loss.append(extract_loss_info(line))

        loss[epoch] = np.array(tmp_loss).mean()
    return loss

def extract_loss_info(line):
    idx = line.find("loss")
    # print(idx)
    return float(line[idx+6:idx+12])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir',default=None,help='working directory missing')
    parser.add_argument(
        '--pretrained', default=None, help='pretrained model file')
    parser.add_argument('--training_dataset',default='VOC07',help='working directory missing')
    args = parser.parse_args()

    os.chdir(args.work_dir)

    for file in glob.glob("*.log"):
        if file.startswith("train"):
            logfile_name = file

    loss = extract_loss_from_log_file(logfile_name)
    pretext_task = extract_task_name(logfile_name)

    plt.figure(1)
    plt.plot(loss, label='backbone training loss, min loss: ' + str(np.min(loss)))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.title(f'Pretrained: {args.pretrained}\nBackbone Training: PT-{pretext_task}, DS: {args.training_dataset}')
    plt.savefig('loss.png')
    # plt.show()
