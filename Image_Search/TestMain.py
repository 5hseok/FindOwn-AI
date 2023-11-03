from Test import CNNModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

root_dir = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\image-05.png"
compare_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\KakaoTalk_20230216_133749847.png"

cnn = CNNModel()
cnn_similarities = cnn.compare_features(target_image_path, 'cnn_features.pkl')

print(cnn_similarities[:10])
image_paths = cnn_similarities[:10]

for i in range(len(image_paths)):
    image_paths[i] = (root_dir + '\\' + image_paths[i][0], image_paths[i][1])
# 서브플롯 생성
fig, ax = plt.subplots(3, 4, figsize=(20, 15))

# 타겟 이미지 표시
img = mpimg.imread(target_image_path)
ax[0, 0].imshow(img)
ax[0, 0].set_title("Target Image")

# 나머지 이미지 표시
for i in range(1, len(image_paths) + 1):
    img = mpimg.imread(image_paths[i-1][0])
    ax[i // 4, i % 4].imshow(img)
    ax[i // 4, i % 4].set_title("Similarity: {:.8f}".format(image_paths[i-1][1]))

# 빈 서브플롯 숨기기
for i in range(len(image_paths) + 1, 12):
    fig.delaxes(ax.flatten()[i])

plt.tight_layout()
plt.show()