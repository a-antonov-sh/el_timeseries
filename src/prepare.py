from data.data import Data

if __name__ == '__main__':
    data = Data()
    data.load_data()
    data.save_prepared()