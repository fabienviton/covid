import keras
import numpy as np
from sklearn import metrics


def print_metrics_binary(y_true, predictions, verbose=1, keepPCR=False):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    # print("predictions", predictions)
    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    if cf.shape[0] == 1:
        tmp00 = cf[0][0]
        cf = np.zeros((2, 2))
        cf[0][0] = tmp00
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    fpr, tpr, thresholds_roc = metrics.roc_curve(y_true, predictions[:, 1])

    (precisions, recalls, thresholds_prc) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))


class HospitalisationMetrics(keras.callbacks.Callback):
    def __init__(self, train_data_x, train_data_y, val_data_x, val_data_y, batch_size=32, verbose=2):
        super(HospitalisationMetrics, self).__init__()
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.val_data_x = val_data_x
        self.val_data_y = val_data_y
        self.verbose = verbose
        self.batch_size = batch_size
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data_x, data_y, history, dataset, logs):
        predictions = self.model.predict(data_x, batch_size=self.batch_size)
        y_true = data_y


        predictions = np.array(predictions)
        predictions_flat = np.reshape(predictions, (-1, 1))
        predictions_flat = np.stack([1 - predictions_flat, predictions_flat], axis=1)
        print("predictions_flat.shape", predictions_flat.shape)

        ret = print_metrics_binary(y_true, predictions_flat, )

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data_x, self.train_data_y, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data_x, self.val_data_y, self.val_history, 'val', logs)

