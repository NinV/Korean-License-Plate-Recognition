## Korean car license plate recognition using LPRNet
This repository is based on the paper  [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/pdf/1806.10447.pdf). 

We use the [KarPlate Dataset](http://pr.gachon.ac.kr/ALPR.html) for training and test model

## Dependencies
- Python 3.6+
- Tensorflow 1.15 or 2
- Opencv 4
- tqdm
- editdistance

## Usage
Unzip the best_weights.zip and data.zip for trained model and dataset
### Training
```bash
python train.py -l data/label.json -i data/train_images --valid_label data/test.json --valid_img_dir data/test_images --save_weights_only --load_all 
```

### Testing
```bash
python predict.py -i data/test_images/4.jpg -w best_weights/weights_best.pb
```

### Evaluate
```bash
python evaluate.py -l data/test.json -i data/test_images/ -w best_weights/weights_best.pb
```

