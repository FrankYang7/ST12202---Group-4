import os 
with open('test.txt','w') as f:
    after_generate1 = os.listdir("./processed/Test/Mask")
    for image in after_generate1:
    	if image.endswith("jpg"):
        	f.write(image + ";" + "0" + "\n")

with open('train.txt','a') as f:
    after_generate2 = os.listdir("./processed/Test/Non Mask")
    for image in after_generate2:
    	if image.endswith("jpg"):
        	f.write(image + ";" + "1" + "\n")