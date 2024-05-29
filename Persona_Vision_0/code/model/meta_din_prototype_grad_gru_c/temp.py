def test(return_grad=True):
    x = 1
    y = 2
    return x, y if return_grad else x


print(test(True))
print(test(False))