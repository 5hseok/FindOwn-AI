from Test import CNNModel
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import numpy as np

cnn = CNNModel()
root_dir = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\image-05.png"
compare_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\KakaoTalk_20230216_133749847.png"

# cnn_similarities = cnn.compare_features(target_image_path, 'cnn_features.pkl')

# print(cnn_similarities[:10])
# image_paths = cnn_similarities[:10]

# for i in range(len(image_paths)):
#     image_paths[i] = (root_dir + '\\' + image_paths[i][0], image_paths[i][1])
# # 서브플롯 생성
# fig, ax = plt.subplots(3, 4, figsize=(20, 15))

# # 타겟 이미지 표시
# img = mpimg.imread(target_image_path)
# ax[0, 0].imshow(img)
# ax[0, 0].set_title("Target Image")

# # 나머지 이미지 표시
# for i in range(1, len(image_paths) + 1):
#     img = mpimg.imread(image_paths[i-1][0])
#     ax[i // 4, i % 4].imshow(img)
#     ax[i // 4, i % 4].set_title("Similarity: {:.8f}".format(image_paths[i-1][1]))

# # 빈 서브플롯 숨기기
# for i in range(len(image_paths) + 1, 12):
#     fig.delaxes(ax.flatten()[i])

# plt.tight_layout()
# plt.show()

from Test import LogoColorSimilarityModel

def main():
    # 로고 이미지들이 저장된 디렉토리
    root_dir = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"
    target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\image-05.png"
    compare_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\KakaoTalk_20230216_133749847.png"
    # 히스토그램을 저장할 파일 경로
    save_path = 'Testhistograms.pkl'

    # LogoColorSimilarityModel 클래스 초기화
    model = LogoColorSimilarityModel(num_bins=30, resize_shape=(256, 256), num_colors=3)

    # 로고 이미지들의 히스토그램 계산 및 저장
    # model.save_histograms(root_dir, save_path)

    # 저장된 히스토그램 불러오기
    histograms = model.load_histograms(save_path)

    # 대상 로고 이미지와 저장된 히스토그램들을 비교하여 유사도 계산
    similarities = model.predict(target_image_path, histograms)

    # 유사도가 낮은 순으로 이미지 파일명과 유사도 출력
    for filename, similarity in similarities:
        print(f"{filename}: {similarity}")

if __name__ == "__main__":
    main()