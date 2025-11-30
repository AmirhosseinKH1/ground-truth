'''
HELLo World


'''


import numpy as np
import cv2
import os

class grand_truth():
    def __init__(self, path):
        self.path = path
    

    def paths(self,):
        label_names = []
        for folders, subfolders, filenames in os.walk(self.path):
            for label in filenames:
                label_names.append(folders + '\\' +label)
        return label_names

  
    def label_reader(self, path):
        labels = []
        for address in path:
            with open(address, 'r') as f:
                lines = f.readlines()
                boxes = []
                for line in lines:
                    line.strip()
                    box = line.split()[1:5]
                    c = int(float(line.split()[0]))
                    x = int(float(box[0]) * 640)
                    y = int(float(box[1]) * 640)
                    w = int(float(box[2]) * 640)
                    h = int(float(box[3]) * 640)
                    # print(c)
                    boxes.append([x, y, w, h, c])
            labels.append(np.array(boxes))
        a = np.array(labels, dtype=object)
        return a
        

    def show(self, labels, points):    
        image = cv2.imread(R'.\\image_n\\image0.jpg')
        for i in labels:
            x, y, w, h = i
            x = x - w//2
            y = y - h//2
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), -1)
        cv2.imshow('i', image)
        cv2.waitKey()


    def find_best(self, box1, x2, y2, H):
        anchor = np.array([[0.9, 0.4],
                        [0.45, 0.45],
                        [0.8, 0.8],
                        [0.4, 0.9]])
        anchor *= 1.5
        h_1 = 300 / H
        IoU = np.zeros((4, len(box1)))
        x1, y1, w1, h1 = box1.T
        px2 = x2 * h_1
        py2 = y2 * h_1
        x1 = x1 - (w1 / 2)
        y1 = y1 - (h1 / 2)
        for i in range(4):
            w2, h2 = anchor[i]
            w2 *= h_1
            h2 *= h_1
            x2 = px2 - (w2 / 2)
            y2 = py2 - (h2 / 2)
            wT = (np.minimum(x2+w2, x1+w1) - np.maximum(x2, x1))
            hT = (np.minimum(y2+h2, y1+h1) - np.maximum(y2, y1))
            shared_area = wT * hT
            total_area = (h1 * w1) + (h2 * w2) - shared_area
            IoU[i] = (shared_area / total_area)
        return np.argmax(IoU.T, axis=0)    


    def make_grand_truth(self, labels, type, nmb_classes):
        grand_truths = []
        for i in range(len(labels)):
            grand_truth = np.zeros((type, type, 4 * (nmb_classes + 5)))
            x, y, w, h = labels[i][:, 0:4].T
            h_1 = 300 / type
            x_g = x // h_1
            y_g = y // h_1
            pcs = self.find_best(labels[i][:, 0:4], x_g + 0.5, y_g + 0.5, type)
            c = labels[i][:, 4]
            anchor_type = pcs * (type + 5)
            x_g = x_g.astype('uint16')
            y_g = y_g.astype('uint16')
            for j in range(len(x_g)):
                grand_truth[y_g[j], x_g[j], anchor_type] = 1
                grand_truth[y_g[j], x_g[j], anchor_type + 1] = labels[i][j, 0]
                grand_truth[y_g[j], x_g[j], anchor_type + 2] = labels[i][j, 1]
                grand_truth[y_g[j], x_g[j], anchor_type + 3] = labels[i][j, 2]
                grand_truth[y_g[j], x_g[j], anchor_type + 4] = labels[i][j, 3]
                grand_truth[y_g[j], x_g[j], anchor_type + 5 + c[j]] = 1
            grand_truths.append(grand_truth)
        return np.array(grand_truths)
    

def run():
    path_label = R'C:\Users\Amir\Desktop\ssd\grand_truth\label_n'
    gt = grand_truth(path_label)
    lb_paths = gt.paths()
    labels = gt.label_reader(lb_paths)

    grand_truth_19 = gt.make_grand_truth(labels, 19, 51)
    grand_truth_10 = gt.make_grand_truth(labels, 10, 51)
    grand_truth_5 = gt.make_grand_truth(labels, 5, 51)

    print('DONE')
if __name__ == '__main__':

    run()
