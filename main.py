from data_preparation import data_preparation
from random_forests import train
from training_model import train_linea_regression

def main():
    data_preparation()
    train()
    train_linea_regression()

if __name__=="__main__":
    main()
