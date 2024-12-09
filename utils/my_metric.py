import keras.backend as K


def get_accuracy(y_true, y_pred):
        # y_true=1, y_pred=1
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # y_true=0, y_pred=1
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    # y_true=0, y_pred=0
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    # y_true=1, y_pred=0
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return (tp + tn)/(tp + tn + fp + fn + K.epsilon())

def get_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def get_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (positives + K.epsilon())
    return recall


def get_F_score(y_true, y_pred, beta=0.5):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    p = get_precision(y_true, y_pred)
    r = get_recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def get_MCC(y_true, y_pred):
    # y_true=1, y_pred=1
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # y_true=0, y_pred=1
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    # y_true=0, y_pred=0
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    # y_true=1, y_pred=0
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    # print(tp, tn, fp, fn)
    return (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + K.epsilon()) ** 0.5

def get_tp(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return tp

def get_fp(y_true, y_pred):
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return fp

def get_tn(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    return tn

def get_fn(y_true, y_pred):
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return fn

def get_metrics(true_positive: int, false_positive: int, true_negative: int, false_negative: int):
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * precision * recall /  (precision + recall)
    MCC = (true_positive * true_negative - false_positive * false_negative) / (
            (true_positive + false_positive) * (true_positive + false_negative) * (
            true_negative + false_positive) * (true_negative + false_negative)) ** 0.5

    return accuracy, precision, recall, f_score, MCC


if __name__ == '__main__':
    pass
