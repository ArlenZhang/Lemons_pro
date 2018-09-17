import re
def do_split():
	with open('./sample','r') as f:
		s=f.read()
	content=re.findall(r'\:\"([\s\S]*?)\"',s,re.M)
	print(content)
	#print(s)
do_split()
