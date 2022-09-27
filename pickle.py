import pickle

if __name__ == '__main__':

    # open a file, where you stored the pickled data
    file = open('dicionarios/counter_twitterTextos_tweets_controle_pandemia.pickle', 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()

    print('Showing the pickled data:')

    cnt = 0
    for item in data:
        print('The data ', cnt, ' is : ', item)
        cnt += 1
