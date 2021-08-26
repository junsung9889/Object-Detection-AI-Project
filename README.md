# Object-Detection-AI-Project

## Explanation of hyperparameters

```
learning rate = 0.001 (float learning rate of data)
num_epochs = 40 (int total epochs to train)
batch_size = 2 (int batch size number)
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4) (optimizer using SGD)
criterion = lossFunction(7,2,5,0.5) (criterion using loss function defined in loss.py)
```

## The command you used to train your model

```
python3 train_2016312160.py
```

## The command you used to test your model

```
python3 test_2016312160.py
```

## Path to your data

```
├── 2016312160_이준성.md (Readme file)
├── TestResults (Result images location of Testing)
├── VOC2012 (VOC2012 Datas)
│   ├── Annotations
│   ├── JPEGImages
│   ├── TestImages (made from Divide_Images.py)
│   ├── TrainImages (made from Divide_Images.py)
│   └── ValidImages (made from Divide_Images.py)
├── model_2016312160.pth (actual model)
├── resnet50-19c8e357.pth (pretrained resnet model)
├── test_2016312160.py
├── train_2016312160.py
├── utils_2016312160
│   ├── Dataset.py
│   ├── Divide_Images.py
│   ├── Loss.py
│   ├── Model.py
│   └── Parse_Annotations.py
├── voc2012_test.txt (made from Parse_Annotations.py)
├── voc2012_train.txt (made from Parse_Annotations.py)
└── voc2012_valid.txt (made from Parse_Annotations.py)
```


## How to Run

In current directory

1. Change directory to utils_2016312160

```
cd ./utils_2016312160
```

2. Run Parse_Annotation.py to make txt files

```
python3 Parse_Annotations.py
```

3. Run Divide_Images.py to divide Images

```
python3 Divide_Images.py
```

4. Move to original directory

```
cd ..
```

5. Train

```
python3 train_2016312160.py
```

6. Test

```
python3 test_2016312160.py
```

7. Finally Check 'TestResults' Folder
