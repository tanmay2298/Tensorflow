class SimpleClass():
	def __init__(self):
		print("Hello")
	def yell(self):
		print("YELLING")
x = SimpleClass()
x.yell()

class ExtendedClass(SimpleClass):
	def __init__(self):
		super().__init__()
		print("Extend")
y = ExtendedClass()