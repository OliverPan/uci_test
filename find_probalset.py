import predict
import copy
import sys


def import_r():
	with open("./data/TestData/TestData/Ra.txt", "r+") as fi:
		ra_set = fi.readlines()
	with open("./data/TestData/TestData/Rb.txt", "r+") as fi:
		rb_set = fi.readlines()
	with open("./data/TestData/TestData/leak0.txt", "r+") as fi:
		leak_set = fi.readlines()
	for i in range(ra_set.__len__()):
		ra_set[i] = ra_set[i][:-1]
		rb_set[i] = rb_set[i][:-1]
		leak_set[i] = leak_set[i][:-1]
	return ra_set, rb_set, leak_set


def expand(data_input):
	length = data_input.__len__()
	print("length:", length)
	output_list = [data_input]
	
	# 一个bit发生变化
	for bit in range(length):
		temp = copy.deepcopy(data_input)
		if temp[bit] == "0":
			temp[bit] = "1"
			output_list.append(temp)
		else:
			temp[bit] = "0"
			output_list.append(temp)
	# print("output_list:", output_list.__len__())
	return output_list


def filter(input_list, label):
	remove_list = []
	for data in input_list:
		flag = False
		for bit in range(4):
			if predict.predict(data, bit) == int(label[bit]):
				continue
			else:
				flag = True
		if flag:
			remove_list.append(data)
			"""
			for bit in range(4):
				print("remove", "".join(data), predict.predict(data, bit), label[bit])
			"""
	for data in remove_list:
		input_list.remove(data)

	for data in input_list:
		for bit in range(4):
			print("".join(data), predict.predict(data, bit), label[bit])


def find_probalset(rc, leak):
	temp_list = expand(list(rc))
	filter(temp_list, leak)
	return temp_list

			
if __name__ == "__main__":
	file_rb = sys.argv[1]
	file_leak = sys.argv[2]
	node = int(sys.argv[3])
	with open(file_rb, "r+") as fi:
		rb_set = fi.readlines()
	with open(file_leak, "r+") as fi:
		leak_set = fi.readlines()
	for i in range(rb_set.__len__()):
		rb_set[i] = rb_set[i][:-1]
		leak_set[i] = leak_set[i][:-1]
	temp_list = find_probalset(rb_set[node], leak_set[node])
	for i in range(temp_list.__len__()):
		temp_list[i] = "".join(temp_list[i])
	"""
	ra_set, rb_set, leak_set = import_r()
	node = 124
	temp_list = find_probalset(rb_set[node], leak_set[node])
	for i in range(temp_list.__len__()):
		temp_list[i] = "".join(temp_list[i])
	print("start:")
	print(temp_list.__len__())
	print(temp_list)
	print(rb_set[node])
	print(ra_set[node])
	if ra_set[node] in temp_list:
		print("true")
	"""