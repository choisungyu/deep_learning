from PIL import Image
#1
# 이미지 열기

im = Image.open('C:/deep_learning/3b/Third_degree_scald_2.jpg')
# 이미지 크기 출력
print(im.size)

# 이미지 JPG로 저장
im.save('C:/deep_learning/3b/Third_degree_scald_2.jpg')

# 이미지 크기 조정

im = Image.open('C:/deep_learning/3b/Third_degree_scald_2.jpg')

size = (100, 100)
im.thumbnail(size)

im.save('C:/deep_learning/3b/Third_degree_scald_1.jpg')

# 이밎 부분 잘라내기(Cropping)
cropImage = im.crop((100, 100, 150, 150))
cropImage.save('C:/deep_learning/3b/Third_degree_scald_1.jpg')


int flag =
which