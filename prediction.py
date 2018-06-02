import torch

if __name__ == "__main__":
    model = torch.load('model.pt')
    input = input('Enter the sentence: ')

