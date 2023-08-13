from model.model import Model

def main():
    model = Model()
    model.train()
    model.test()
    model.save()

if __name__ == '__main__':
    main()