# rough count of N's

f_all = open("GL000219.1.fasta", "r")
all_stuff = f_all.read()

print ("NUmber N values: ", all_stuff.count("N"))
print ("Numbe A values: ", all_stuff.count("-"))
