## **Korean car license plate recognition using LPRNet**
This repository is based on the paper  [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/pdf/1806.10447.pdf). 

We use the [KarPlate Dataset](http://pr.gachon.ac.kr/ALPR.html) for training and test model

## **Dependencies**
- Python 3.6+
- Tensorflow 1.15 or 2
- Opencv 4
- tqdm
- editdistance

## **Usage**
### *Dataset*
Download [data.zip](https://bit.ly/3egQ9jU), unzip and move into **data** folder for training and testing

### *Pre-trained model*
Download [best_weights.zip](https://bit.ly/2zt5hMc), unzip and move into **saved_models** folder for testing

### *Demo*
```bash
python predict.py -i data/test_images/4.jpg -w saved_models/weights_best.pb
```

### *Training*
```bash
python train.py -l data/label.json -i data/train_images --valid_label data/test.json --valid_img_dir data/test_images --save_weights_only --load_all 
```

### *Testing*
```bash
python predict.py -i data/test_images/4.jpg -w saved_models/weights_best.pb
```

### *Evaluate*
```bash
python evaluate.py -l data/test.json -i data/test_images/ -w saved_models/weights_best.pb
```

