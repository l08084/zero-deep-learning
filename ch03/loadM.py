import sys
import os
from mnist import load_mnist
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

# 最初の呼び出しは数分待ちます...
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# それぞれのデータの形状を出力
print(x_train.shape)  # (6000, 784)
