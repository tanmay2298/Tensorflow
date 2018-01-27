import numpy as np
# Operation
class Operation():
	def __init__(self, input_nodes = []):
		self.input_nodes = input_nodes
		self.output_nodes = []

		for node in input_nodes:
			node.output_nodes.append(self)

		_default_graph.operations.append(self)
	def compute(self):
		pass

class add(Operation):
	def __init__(self, x, y):
		super().__init__([x, y])

	def compute(self, x_var, y_var):
		self.inputs = [x_var, y_var]
		return x_var + y_var

class multiply(Operation):
	def __init__(self, x, y):
		super().__init__([x, y])

	def compute(self, x_var, y_var):
		self.inputs = [x_var, y_var]
		return x_var * y_var

class matmul(Operation):
	def __init__(self, x, y):
		super().__init__([x, y])

	def compute(self, x_var, y_var):
		self.inputs = [x_var, y_var]
		return x_var.dot(y_var)

class Placeholder():
	def __init__(self):
		self.output_nodes = []
		_default_graph.placeholders.append(self)

class Variable():
	def __init__(self, initial_value = None):
		self.value = initial_value
		self.output_nodes = []
		_default_graph.variables.append(self)

class Graph():
	def __init__(self):
		self.operations = []
		self.placeholders = []
		self.variables = []

	def set_as_default(self):
		global _default_graph
		_default_graph = self

def traverse_postorder(operation):
	nodes_postorder = []

	def recurse(node):
		if isinstance(node, operation):
			for input_node in node.input_nodes:
				recurse(input_node)
			nodes_postorder.append(node)
		recurse(operation)
	return nodes_postorder
"""
z = Ax + b
A = 10
b = 1
z = 10x + 1
"""
g = Graph()

g.set_as_default()

A = Variable(10)
b = Variable(1)

x = Placeholder()

y = multiply(A, x)

z = add(y, b)

class Session():
	def run(self, operation, feed_dict = {}):
		nodes_postorder = traverse_postorder(operation)

		for node in nodes_postorder:
			if type(node) == Placeholder:
				node.output = feed_dict[node]
			elif type(node) == Variable:
				node.output = node.value
			else:
				node.inputs = [input_node.output for input_node in node.input_node]
				node.output = node.compute(*node.inputs) #args

			if type(node.output) == list:
				node.output = np.array(node.output)
		return operation.output

sess = Session()

result = sess.run(operation = z, feed_dict = {x:10})

print(result)