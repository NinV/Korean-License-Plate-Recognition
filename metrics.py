import numpy as np
import tensorflow as tf
import editdistance
from loader import Loader


def ctc_loss(labels, predicts, input_lengths, label_lengths):
    loss = tf.keras.backend.ctc_batch_cost(
        labels,
        predicts,
        input_lengths,
        label_lengths)
    loss = tf.keras.backend.mean(loss)
    return loss


class Evaluator:
    def __init__(self, model, labelfile, img_dir, batch_size=32):
        self.loader = Loader(labelfile, img_dir, load_all=True, augmen_func=None)
        if len(self.loader) == 0:
            raise ValueError("Empty loader")

        self.model = model
        self.batch_size = batch_size
        self.losses, self.CERs, self.WERs = [], [], []

    def evaluate(self):
        imgs, labels, label_lengths = [], [], []
        self.losses, self.CERs, self.WERs = [], [], []
        last_batch_size = self.batch_size
        for i, (img, label, label_length) in enumerate(self.loader):
            # the loader does not raise StopIteration so we need to manually break the for loop
            # the for loop will catch the StopIteration but not sure about tensorflow data.dataset,
            # so at the moment the Loader does not raise StopIteration to avoid crash
            # TODO: Consider update this in future
            if i == len(self.loader):
                if labels:
                    print("Evaluating batch {}".format(len(self.losses) + 1))
                    self._flush(imgs, labels, label_lengths)
                    last_batch_size = len(labels)
                break
            if (i + 1) % self.batch_size == 0:
                print("Evaluating batch {}".format(len(self.losses) + 1))
                self._flush(imgs, labels, label_lengths)
                imgs, labels, label_lengths = [], [], []

            else:
                imgs.append(img)
                labels.append(label)
                label_lengths.append(label_length)
        loss = self._average(self.losses, last_batch_size)
        cer = self._average(self.CERs, last_batch_size)
        wer = self._average(self.WERs, last_batch_size)
        self._print_result(loss, cer, wer)
        return loss, cer, wer

    def _average(self, values, last_batch_size):
        if len(values) == 1:
            return values[0]
        return (np.sum(values[:-1] * self.batch_size) + values[-1] * last_batch_size) / len(self.loader)

    def _flush(self, imgs, labels, label_lengths):
        imgs = np.array(imgs)
        logits = self.model.model.predict(imgs)
        decoded_texts = self.model.decode_pred(logits, self.loader.class_names)
        loss = self._calc_loss(logits, labels, label_lengths)
        label_texts = self._decode_label(labels)
        CER, WER = self._calc_CER_and_WER(label_texts, decoded_texts)
        print("loss: {} - CER: {}, WER: {}\n".format(float(loss), CER, WER))
        self.losses.append(float(loss))
        self.CERs.append(CER)
        self.WERs.append(WER)

    def _decode_label(self, labels):
        results = []
        for d in labels:
            text = []
            for idx in d:
                if idx == -1:
                    break
                text.append(self.loader.class_names[idx])
            results.append(''.join(text))
        return results

    def _calc_loss(self, logits, labels, label_lengths):
        batch_size, times = logits.shape[:2]
        logits_lengths = tf.expand_dims(tf.tile(tf.constant([times], tf.int32),
                                                tf.constant([batch_size], tf.int32)),
                                        axis=1)
        loss_value = ctc_loss(labels, logits, logits_lengths, label_lengths)
        return loss_value

    def _calc_CER_and_WER(self, label_texts, decoded_texts):
        ed = []
        WER = 0
        for label, pred in zip(label_texts, decoded_texts):
            cer = editdistance.eval(label, pred)
            ed.append(cer)
            if cer != 0:
                WER += 1
        WER /= len(label_texts)
        CER = sum(ed) / len(label_texts)
        return CER, WER

    def _print_result(self, loss, CER, WER):
        print("Number of samples in test set: {}\n"
              "mean loss: {}\n"
              "mean CER: {}\n"
              "WER: {}\n".format(len(self.loader),
                                 loss,
                                 CER,
                                 WER
                                 )
              )
