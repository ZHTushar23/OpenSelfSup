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
    loss = np.zeros([num_epochs])
    acc  = np.zeros([num_epochs])
    top1 = np.zeros([num_epochs])
    top5 = np.zeros([num_epochs])

    for epoch in range(num_epochs):
        tmp_loss = []
        tmp_acc  = []
        keyword1 = "Epoch ["+str(epoch+1)+"]"
        keyword2 = "Epoch(val) ["+str(epoch+1)+"]"

        with open(logfile, 'r') as f:
            for line in f:
                if keyword1 in line:
                    tmp_loss.append(extract_loss_info(line))
                    tmp_acc.append(extract_acc_info(line))

                elif keyword2 in line:
                    top1[epoch] = extract_top1_info(line)
                    top5[epoch] = extract_top5_info(line)

        loss[epoch] = np.array(tmp_loss).mean()
        acc[epoch]  = np.array(tmp_acc).mean()
    return loss, acc, top1, top5

def extract_loss_info(line):
    idx = line.find("loss")
    # print(idx)
    return float(line[idx+6:idx+12])

def extract_acc_info(line):
    idx = line.find("acc")
    # print(idx)
    return float(line[idx+5:idx+11])

def extract_top1_info(line):
    idx = line.find("head0_top1")
    # print(idx)
    return float(line[idx+12:idx+19])

def extract_top5_info(line):
    idx = line.find("head0_top2")
    # print(idx)
    return float(line[idx+12:idx+19])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir',default=None,help='working directory missing')
    args = parser.parse_args()

    os.chdir(args.work_dir)

    for file in glob.glob("*.log"):
        if file.startswith("train"):
            logfile_name = file

    loss, acc, top1, top5 =  extract_info_from_log_file(logfile_name)

    plt.figure(1)
    plt.plot(loss, label='classifier training loss, min loss: ' + str(np.min(loss)))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.title("Benchmark Dataset: ImageNet")
    plt.savefig('loss.png')

    plt.figure(2)
    plt.plot(acc, label='classifier training accuracy, max acc: ' + str(np.max(acc)))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title("Benchmark Dataset: ImageNet")
    plt.savefig('acc.png')

    plt.figure(3)
    plt.plot(top1, label='classifier top1 score, max score: ' + str(np.max(top1)))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title("Benchmark Dataset: ImageNet")
    plt.savefig('top1.png')

    plt.figure(4)
    plt.plot(top5, label='classifier top5 score, max acc: ' + str(np.max(top5)))
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title("Benchmark Dataset: ImageNet")
    plt.savefig('top5.png')


    #plt.show()
