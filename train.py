from api.yf import getStocksDF

def main():
    data = getStocksDF()
    print(data)

if __name__ == '__main__':
    print('__main__ start')
    main()
    print('__main__ end')
