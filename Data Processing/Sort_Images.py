import shutil

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

index = 0
for l in labels:
 count = 0
 for i in range(index, index+450):
  shutil.move("./dataset/left/"+str(i)+".jpg", "./Image_Directory/"+l+"/"+str(count)+".jpg")
  count += 1
 for i in range(index, index+450):
  shutil.move("./dataset/right/"+str(i)+".jpg", "./Image_Directory/"+l+"/"+str(count)+".jpg")
  count += 1
 index += 450
