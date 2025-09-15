# Animal Classifier Model
=====================Project Directory Struncture=====================
#### Step-1: Organize your project directory like this:
```
src/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   │   ├── cat1.jpg
│   │   │   ├── cat2.jpg
│   │   │   └── ...
│   │   ├── cows/
│   │   │   ├── cow1.jpg
│   │   │   ├── cow2.jpg
│   │   │   └── ...
│   │   ├── dogs/
│   │   │   ├── dog1.jpg
│   │   │   ├── dog2.jpg
│   │   │   └── ...
│   │   └── unknown/
│   │       ├── unknown1.jpg
│   │       ├── unknown2.jpg
│   │       └── ...
│   ├── validation/
│       ├── cats/
│       │   ├── cat1.jpg
|       |   |__ ...
│       |── cows/
|       |   ├── cow1.jpg
|       |   |__ ...
│       ├── dogs/
│       │   ├── dog1.jpg
|       |   |__ ...
│       └── unknown/
│           ├── unknown1.jpg
|           |__ ...
│
├── models/
│   └── animal_model.keras // this will be generated after you run train.py
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```
#### Step-2: Set Up Your Virtual Environment
##### 1.  Create a virtual environment (optional but recommended):
```
python -m venv venv
```
###### Activate the environment:

* Windows: .\venv\Scripts\activate
* Mac/Linux: source venv/bin/activate

##### 2. Install the required packages:
```
pip install -r requirements.txt
```

#### Step-3: Train the Model
##### 1. Run the train.py script to train the model:
```
python scripts/train.py
```
This script will train the model using the training data and save the model to the models directory.

#### Step-4: Evaluate the Model
##### 1. Run the evaluate.py script to evaluate the model:
```
python scripts/evaluate.py
```
This script will evaluate the model using the validation data and print out the accuracy and other metrics.

#### Step-5: Make Predictions
##### 1. Run the predict.py script to make predictions on new data:
```
python scripts/predict.py path/to/your/image.jpg
```
This script will load the trained model and make predictions on the image at the specified path.



