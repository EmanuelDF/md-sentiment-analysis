import pandas as pd

if __name__ == '__main__':
    train = pd.read_excel('resource/model.xlsx')
    # test  = pd.read_csv('resource/test.csv')
    train['text'] = train['text'].apply(str)
    train['sentiment'] = train['sentiment'].apply(str)
    print(train.head())

    from pycaret.classification import *

    s = setup(train, target='sentiment')

    best = compare_models()
    evaluate_model(best)

    # predictions = predict_model(best, data = test)

    # save_model(best, 'actual_model')
