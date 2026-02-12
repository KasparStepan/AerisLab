
class Test:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    @property
    def price(self):
        return self._price    

    @price.setter
    def price(self, new_price):
        if new_price < 0:
            raise ValueError("Price cannot be negatice. Please enter new")
        else:
            self._price = new_price

if __name__ == "__main__":
    test = Test("Sample Test", 100)
    print(f"Initial Price: {test._price}")
    try:
        test.price = -50
    except ValueError as e:
        print(e)
    test.price = 150
    print(f"Updated Price: {test._price}")