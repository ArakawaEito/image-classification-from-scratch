import numpy as np
from collections import OrderedDict

""" functions """
def cross_entropy_error(y, t):
    """
    y : 出力値(通常は、0-1の確率)  
    t : 正解値(通常は、0or1)  
    """
    if y.ndim==1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
        
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum( t * np.log(y + delta)) / batch_size

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def im2col(input_data, filter_h, filter_w, stride=1, pad=0, constant_values=0):
    """
    input_data : (データ数, チャンネル数, 高さ, 幅)の4次元配列からなる入力データ. 画像データの形式を想定している
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    constant_values : パディング処理で埋める際の値
    return : 2次元配列
    """
    
    # 入力データのデータ数, チャンネル数, 高さ, 幅を取得する
    N, C, H, W = input_data.shape 
    
    # 出力データ(畳み込みまたはプーリングの演算後)の形状を計算する
    out_h = (H + 2*pad - filter_h)//stride + 1 # 出力データの高さ(端数は切り捨てる)
    out_w = (W + 2*pad - filter_w)//stride + 1 # 出力データの幅(端数は切り捨てる)

    # パディング処理
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)],
                             'constant', constant_values=constant_values) # pad=1以上の場合、周囲を0で埋める
    
    # 配列の初期化
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) 

    # 配列を並び替える(フィルター内のある1要素に対応する画像中の画素を取り出してcolに代入する)
    for y in range(filter_h):
        """
        フィルターの高さ方向のループ
        """
        y_max = y + stride*out_h
        
        for x in range(filter_w):
            """
            フィルターの幅方向のループ
            """
            x_max = x + stride*out_w
            
            # imgから値を取り出し、colに入れる
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # y:y_max:strideの意味  :  yからy_maxまでの場所をstride刻みで指定している
            # x:x_max:stride の意味  :  xからx_maxまでの場所をstride刻みで指定している

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) # 軸を入れ替えて、2次元配列(行列)に変換する
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0, is_backward=False):
    """
    Parameters
    ----------
    col : 2次元配列
    input_shape : 入力データの形状,  (データ数, チャンネル数, 高さ, 幅)の4次元配列
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    return : (データ数, チャンネル数, 高さ, 幅)の4次元配列. 画像データの形式を想定している
    -------
    """
    
    # 入力画像(元画像)のデータ数, チャンネル数, 高さ, 幅を取得する
    N, C, H, W = input_shape
    
    # 出力(畳み込みまたはプーリングの演算後)の形状を計算する
    out_h = (H + 2*pad - filter_h)//stride + 1 # 出力画像の高さ(端数は切り捨てる)
    out_w = (W + 2*pad - filter_w)//stride + 1 # 出力画像の幅(端数は切り捨てる)
    
    # 配列の形を変えて、軸を入れ替える
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 配列の初期化
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))  # pad分を大きくとる. stride分も大きくとる
    
    # 配列を並び替える
    for y in range(filter_h):
        """
        フィルターの高さ方向のループ
        """        
        y_max = y + stride*out_h
        for x in range(filter_w):
            """
            フィルターの幅方向のループ
            """            
            x_max = x + stride*out_w
            
            # colから値を取り出し、imgに入れる
            if is_backward:
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            else:
                img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]
                
    return img[:, :, pad:H + pad, pad:W + pad] # pad分は除いておく(pad分を除いて真ん中だけを取り出す)

######################################################################################################################################

""" Layers """
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class BatchNormalization:
    def __init__(self, gamma, beta, rho=0.9, moving_mean=None, moving_var=None):
        self.gamma = gamma # スケールさせるためのパラメータ, 学習によって更新させる.
        self.beta = beta # シフトさせるためのパラメータ, 学習によって更新させる
        self.rho = rho # 移動平均を算出する際に使用する係数

        # 予測時に使用する平均と分散
        self.moving_mean = moving_mean # muの移動平均
        self.moving_var = moving_var        # varの移動平均
        
        # 計算中に算出される値を保持しておく変数群
        self.batch_size = None
        self.x_mu = None
        self.x_std = None        
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        """
        順伝播計算
        x :  CNNの場合は4次元、全結合層の場合は2次元  
        """
        if x.ndim == 4:
            """
            画像形式の場合
            """
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1) # NHWCに入れ替え
            x = x.reshape(N*H*W, C) # (N*H*W,C)の2次元配列に変換
            out = self.__forward(x, train_flg)
            out = out.reshape(N, H, W, C)# 4次元配列に変換
            out = out.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif x.ndim == 2:
            """
            画像形式以外の場合
            """
            out = self.__forward(x, train_flg)           
            
        return out
            
    def __forward(self, x, train_flg, epsilon=1e-8):
        """
        x : 入力. n×dの行列. nはあるミニバッチのバッチサイズ. dは手前の層のノード数
        """
        if (self.moving_mean is None) or (self.moving_var is None):
            N, D = x.shape
            self.moving_mean = np.zeros(D)
            self.moving_var = np.zeros(D)
                        
        if train_flg:
            """
            学習時
            """
            # 入力xについて、nの方向に平均値を算出. 
            mu = np.mean(x, axis=0) # 要素数d個のベクトル
            
            # 入力xから平均値を引く
            x_mu = x - mu   # n*d行列
            
            # 入力xの分散を求める
            var = np.mean(x_mu**2, axis=0)  # 要素数d個のベクトル
            
            # 入力xの標準偏差を求める(epsilonを足してから標準偏差を求める)
            std = np.sqrt(var + epsilon)  # 要素数d個のベクトル
            
            # 標準化
            x_std = x_mu / std  # n*d行列
            
            # 値を保持しておく
            self.batch_size = x.shape[0]
            self.x_mu = x_mu
            self.x_std = x_std
            self.std = std
            self.moving_mean = self.rho * self.moving_mean + (1-self.rho) * mu
            self.moving_var = self.rho * self.moving_var + (1-self.rho) * var            
        else:
            """
            予測時
            """
            x_mu = x - self.moving_mean # n*d行列
            x_std = x_mu / np.sqrt(self.moving_var + epsilon) # n*d行列
            
        # gammaでスケールし、betaでシフトさせる
        out = self.gamma * x_std + self.beta # n*d行列
        return out

    def backward(self, dout):
        """
        逆伝播計算
        dout : CNNの場合は4次元、全結合層の場合は2次元  
        """
        if dout.ndim == 4:
            """
            画像形式の場合
            """            
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1) # NHWCに入れ替え
            dout = dout.reshape(N*H*W, C) # (N*H*W,C)の2次元配列に変換
            dx = self.__backward(dout)
            dx = dx.reshape(N, H, W, C)# 4次元配列に変換
            dx = dx.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif dout.ndim == 2:
            """
            画像形式以外の場合
            """
            dx = self.__backward(dout)

        return dx

    def __backward(self, dout):
        # betaの勾配
        dbeta = np.sum(dout, axis=0)
        
        # gammaの勾配(n方向に合計)
        dgamma = np.sum(self.x_std * dout, axis=0)
        # Xstdの勾配
        a1 = self.gamma * dout
        # Xmuの勾配(1つ目)
        a2 = a1 / self.std
        # 標準偏差の逆数の勾配(n方向に合計)
        a3 = np.sum(a1 * self.x_mu, axis=0)
        # 標準偏差の勾配
        a4 = -(a3) / (self.std * self.std)
        # 分散の勾配
        a5 = 0.5 * a4 / self.std
        # Xmuの2乗の勾配
        a6 = a5 / self.batch_size
        # Xmuの勾配(2つ目)
        a7 = 2.0  * self.x_mu * a6
        # muの勾配
        a8 = np.sum(-(a2+a7), axis=0)
        # Xの勾配
        dx = a2 + a7 +  a8 / self.batch_size # 第3項はn方向に平均
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
    
# 畳み込み用アフィン
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応(画像形式のxに対応させる)
        self.original_x_shape = x.shape
        
        x = x.reshape(x.shape[0], -1)
        self.x = x

        # print(self.W.shape)
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        
        # 初期値
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        """
        順伝播
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        """
        dout=1は、他のレイヤと同じ使い方ができるように設定しているダミー変数
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

# 畳み込み層
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # フィルターの重み(配列形状:フィルターの枚数, チャンネル数, フィルターの高さ, フィルターの幅)
        self.b = b #フィルターのバイアス
        self.stride = stride # ストライド数
        self.pad = pad # パディング数
        
        # インスタンス変数の宣言
        self.x = None   
        self.col = None
        self.col_W = None
        self.dcol = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        順伝播計算
        x : 入力(配列形状=(データ数, チャンネル数, 高さ, 幅))
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1 # 出力の高さ(端数は切り捨てる)
        out_w =(W + 2*self.pad - FW) // self.stride + 1# 出力の幅(端数は切り捨てる)

        # 畳み込み演算を効率的に行えるようにするため、入力xを行列colに変換する
        col = im2col(x, FH, FW, self.stride, self.pad)
        
        # 重みフィルターを2次元配列に変換する
        # col_Wの配列形状は、(C*FH*FW, フィルター枚数)
        col_W = self.W.reshape(FN, -1).T

        # 行列の積を計算し、バイアスを足す
        out = np.dot(col, col_W) + self.b
        
        # 画像形式に戻して、チャンネルの軸を2番目に移動させる
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        逆伝播計算
        dout : 出力層側の勾配
        return : 入力層側へ伝える勾配
        """
        FN, C, FH, FW = self.W.shape
        
        # doutのチャンネル数軸を4番目に移動させ、2次元配列に変換する
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # バイアスbはデータ数方向に総和をとる
        self.db = np.sum(dout, axis=0)
        
        # 重みWは、入力である行列colと行列doutの積になる
        self.dW = np.dot(self.col.T, dout)
        
        # (フィルター数, チャンネル数, フィルター高さ、フィルター幅)の配列形状に戻す
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 入力側の勾配は、doutにフィルターの重みを掛けて求める
        dcol = np.dot(dout, self.col_W.T)
        
        # 勾配を4次元配列(データ数, チャンネル数, 高さ, 幅)に変換する
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad, is_backward=True)

        self.dcol = dcol # 結果を確認するために保持しておく
            
        return dx
    
class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):

        self.pool_h = pool_h # プーリングを適応する領域の高さ
        self.pool_w = pool_w # プーリングを適応する領域の幅
        self.stride = stride # ストライド数
        self.pad = pad # パディング数

        # インスタンス変数の宣言
        self.x = None
        self.arg_max = None
        self.col = None
        self.dcol = None
        
            
    def forward(self, x):
        """
        順伝播計算
        x : 入力(配列形状=(データ数, チャンネル数, 高さ, 幅))
        """        
        N, C, H, W = x.shape
        
        # 出力サイズ
        out_h = (H  + 2*self.pad - self.pool_h) // self.stride + 1 # 出力の高さ(端数は切り捨てる)
        out_w = (W + 2*self.pad - self.pool_w) // self.stride + 1# 出力の幅(端数は切り捨てる)    
        
        # プーリング演算を効率的に行えるようにするため、2次元配列に変換する
        # パディングする値は、マイナスの無限大にしておく
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad, constant_values=-np.inf)
        
        # チャンネル方向のデータが横に並んでいるので、縦に並べ替える
        # 変換後のcolの配列形状は、(N*C*out_h*out_w, H*W)になる 
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 最大値のインデックスを求める
        # この結果は、逆伝播計算時に用いる
        arg_max = np.argmax(col, axis=1)
        
        # 最大値を求める
        out = np.max(col, axis=1)
        
        # 画像形式に戻して、チャンネルの軸を2番目に移動させる
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        逆伝播計算
        順伝播計算時に最大値となった場所は、self.arg_maxに保持されている        
        dout : 出力層側の勾配
        return : 入力層側へ伝える勾配
        """        
        
        # doutのチャンネル数軸を4番目に移動させる
        dout = dout.transpose(0, 2, 3, 1)
        
        # プーリング適応領域の要素数(プーリング適応領域の高さ × プーリング適応領域の幅)
        pool_size = self.pool_h * self.pool_w
        
        # 勾配を入れる配列を初期化する
        # dcolの配列形状 : (doutの全要素数, プーリング適応領域の要素数) 
        # doutの全要素数は、dout.size で取得できる
        dcol = np.zeros((dout.size, pool_size))
        
        # 順伝播計算時に最大値となった場所に、doutを配置する
        dcol[np.arange(dcol.shape[0]), self.arg_max] = dout.flatten()
        
        # 勾配を4次元配列(データ数, チャンネル数, 高さ, 幅)に変換する
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad, is_backward=True)
        
        self.dcol = dcol # 結果を確認するために保持しておく
        
        return dx
    
    
def numerical_gradient(f, W):
    """
    全ての次元について、個々の次元だけの微分を求める
    f : 関数
    W : 偏微分を求めたい場所の座標。多次元配列
    """
    h = 1e-4 # 0.0001
    grad = np.zeros_like(W)
    
    it = np.nditer(W, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = W[idx]
        
        W[idx] = tmp_val + h
        fxh1 = f(W)
        
        W[idx] = tmp_val - h 
        fxh2 = f(W)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        W[idx] = tmp_val # 値を元に戻す
        
        # 次のindexへ進める
        it.iternext()   
        
    return grad

##############################################################################################################################

""" Model """
class CustomNN:
    def __init__(self, input_dim=(1, 28, 28), weight_decay_lambda=0.01, weight_init_std=0.01):
        """
        input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : dict, 畳み込みの条件
        pool_param : dict, プーリングの条件
        hidden_size : int, 隠れ層のノード数
        output_size : int, 出力層のノード数
        weight_decay_lambda : Weight Decay（L2ノルム）の強さ
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        """
        self.weight_decay_lambda = weight_decay_lambda
        self.current_size = input_dim[1] # 現時点のデータサイズ(HとWが等しいことを前提)
        self.current_dim = input_dim[0] # 現時点のチャネル数
        self.current_pixel = self.current_dim * self.current_size * self.current_size # 現時点のデータのピクセル数(要素数)
        self.current_layer_num = 0 # 現在のレイヤ数(重み更新のインデックスに使用)

        # 重みの初期化
        self.params = {}
        self.std = weight_init_std

        # レイヤの生成
        self.trainLayers = [] # 学習対象のパラメータを持つ層が何番目の層なのかを記録するリスト
        self.layers = OrderedDict()
        print("-"*50)
        print(f"Layer\t\t\tOutput Shape")
        print("="*50)
        print(f"Input\t\t\t[None, {self.current_dim}, {self.current_size}, {self.current_size}]")
        print("-"*50)

        ## 以下にレイヤを追加 #################################
        self.Conv(input_dim=self.current_dim, filter_num=16, filter_size=3, pad=1, stride=1)
        self.BatchNorm4D(rho=0.9, moving_mean=None, moving_var=None)
        self.Relu()

        self.Conv(input_dim=self.current_dim, filter_num=16, filter_size=3, pad=1, stride=1)
        self.BatchNorm4D(rho=0.9, moving_mean=None, moving_var=None)
        self.Relu()
        self.MaxPool(pool_size=2, stride=2, pad=0)


        self.Conv(input_dim=self.current_dim, filter_num=32, filter_size=3, pad=1, stride=1)
        self.BatchNorm4D(rho=0.9, moving_mean=None, moving_var=None)
        self.Relu()

        self.Conv(input_dim=self.current_dim, filter_num=32, filter_size=3, pad=1, stride=1)
        self.BatchNorm4D(rho=0.9, moving_mean=None, moving_var=None)
        self.Relu()
        self.MaxPool(pool_size=2, stride=2, pad=0)

        self.Conv(input_dim=self.current_dim, filter_num=64, filter_size=3, pad=1, stride=1)
        self.BatchNorm4D(rho=0.9, moving_mean=None, moving_var=None)
        self.Relu()

        self.Conv(input_dim=self.current_dim, filter_num=64, filter_size=3, pad=1, stride=1)
        self.BatchNorm4D(rho=0.9, moving_mean=None, moving_var=None)
        self.Relu()
        self.MaxPool(pool_size=2, stride=2, pad=0)

        self.Liner(input_size=self.current_pixel, output_size=100)
        self.BatchNorm2D(rho=0.9, moving_mean=None, moving_var=None)
        self.DropOut(dropout_ration=0.3)
        self.Relu()

        self.Liner(input_size=self.current_pixel, output_size=10)
        self.Relu()

        self.last_layer = SoftmaxWithLoss()

        # print("params:")
        # print(self.params.keys())
        # print("layers :")
        # print(self.layers.keys() )
        # print("self.trainLayers:")
        # print(self.trainLayers)


    def Conv(self, input_dim, filter_num, filter_size=5, pad=0, stride=1):
        """
        input_dim:入力データのチャネル数
        """
       
        self.std = np.sqrt(2/self.current_pixel) # heの初期値
        
        # self.current_layer_num:何番目のレイヤの重みなのかを格納しておく
        # self.W_params.append([self.std * np.random.randn(filter_num, input_dim, filter_size, filter_size), self.current_layer_num])
        # self.b_params.append([np.zeros(filter_num), self.current_layer_num]) #b1は畳み込みフィルターのバイアスになる

        self.params["W"+str(self.current_layer_num)] = self.std * np.random.randn(filter_num, input_dim, filter_size, filter_size)
        self.params["b"+str(self.current_layer_num)] = np.zeros(filter_num)
        self.layers["Conv"+str(self.current_layer_num)] = Convolution(self.params["W"+str(self.current_layer_num)] , self.params["b"+str(self.current_layer_num)], stride, pad)
        self.trainLayers.append(self.current_layer_num)

        self.current_size = (self.current_size + 2*pad - filter_size) // stride + 1 # 畳み込み後のサイズ(H,W共通)
        self.current_dim = filter_num
        self.current_pixel = self.current_dim * self.current_size * self.current_size # 畳み込み後のピクセル総数

        print(f"Conv{self.current_layer_num}\t\t\t[None, {self.current_dim}, {self.current_size}, {self.current_size}]")
        print("-"*50)

        self.current_layer_num += 1
        
    def MaxPool(self, pool_size=2, stride=2, pad=0):
        self.current_size = (self.current_size + 2*pad - pool_size) // stride + 1 # プーリング後のサイズ(H,W共通)
        # print("maxpool", self.current_size)
        self.current_pixel = self.current_dim * self.current_size * self.current_size # プーリング後のピクセル総数      
        # self.layers.append(MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=stride, pad=pad))  
        self.layers["MaxPool"+str(self.current_layer_num)] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=stride, pad=pad)
        
        print(f"MaxPool{self.current_layer_num}\t\t[None, {self.current_dim}, {self.current_size}, {self.current_size}]")
        print("-"*50)
        
        self.current_layer_num += 1
    
    def Liner(self, input_size, output_size):
        self.std = np.sqrt(2/self.current_pixel) # heの初期値

        # self.W_params.append([self.std *  np.random.randn(input_size, output_size), self.current_layer_num])
        # self.b_params.append([np.zeros(output_size), self.current_layer_num])
        # self.layers.append(Affine(self.W_params[-1][0], self.b_params[-1][0]))  

        self.params["W"+str(self.current_layer_num)] = self.std *  np.random.randn(input_size, output_size)
        self.params["b"+str(self.current_layer_num)] = np.zeros(output_size)
        self.layers["Affine"+str(self.current_layer_num)] = Affine(self.params["W"+str(self.current_layer_num)] , self.params["b"+str(self.current_layer_num)])
        self.trainLayers.append(self.current_layer_num)

        self.current_dim = 0
        self.current_pixel = output_size

        print(f"Affine{self.current_layer_num}\t\t[None, {self.current_pixel}]")
        print("-"*50)

        self.current_layer_num += 1

    def BatchNorm2D(self, rho=0.9, moving_mean=None, moving_var=None):
        """
        入力が2次元の場合
        """
        gamma = np.ones(self.current_pixel)
        beta = np.zeros(self.current_pixel)
        # self.W_params.append([gamma, self.current_layer_num])
        # self.b_params.append([beta , self.current_layer_num])
        # self.layers.append(BatchNormalization(gamma, beta, rho=rho, moving_mean=moving_mean, moving_var=moving_var))  

        self.params["W"+str(self.current_layer_num)] = gamma
        self.params["b"+str(self.current_layer_num)] = beta

        self.params["MM"+str(self.current_layer_num)] = moving_mean
        self.params["MV"+str(self.current_layer_num)] = moving_var

        self.layers["BatchNorm2D"+str(self.current_layer_num)] = BatchNormalization(gamma, beta, rho=rho, moving_mean=moving_mean, moving_var=moving_var)
        self.trainLayers.append(self.current_layer_num)

        print(f"BatchNorm2D{self.current_layer_num}\t\t[None, {self.current_pixel}]")
        print("-"*50)

        self.current_layer_num += 1

    def BatchNorm4D(self, rho=0.9, moving_mean=None, moving_var=None):
        """
        入力が4次元の場合
        """
        gamma = np.ones(self.current_dim)
        beta = np.zeros(self.current_dim)
        # self.W_params.append([gamma, self.current_layer_num])
        # self.b_params.append([beta , self.current_layer_num])
        # self.layers.append(BatchNormalization(gamma, beta, rho=rho, moving_mean=moving_mean, moving_var=moving_var))  

        self.params["W"+str(self.current_layer_num)] = gamma
        self.params["b"+str(self.current_layer_num)] = beta

        self.params["MM"+str(self.current_layer_num)] = moving_mean
        self.params["MV"+str(self.current_layer_num)] = moving_var

        self.layers["BatchNorm4D"+str(self.current_layer_num)] = BatchNormalization(gamma, beta, rho=rho, moving_mean=moving_mean, moving_var=moving_var)
        self.trainLayers.append(self.current_layer_num)

        print(f"BatchNorm4D{self.current_layer_num}\t\t[None, {self.current_dim}, {self.current_size}, {self.current_size}]")
        print("-"*50)

        self.current_layer_num += 1

    def Relu(self):
        # self.layers.append(ReLU())  
        self.layers["Relu"+str(self.current_layer_num)] = ReLU()

        if self.current_dim==0:
            print(f"Relu{self.current_layer_num}\t\t\t[None, {self.current_pixel}]")
            print("-"*50)
        else:
            print(f"Relu{self.current_layer_num}\t\t\t[None, {self.current_dim}, {self.current_size}, {self.current_size}]")
            print("-"*50)

        self.current_layer_num += 1

    def DropOut(self, dropout_ration):
        self.layers["Dropout"+str(self.current_layer_num)] = Dropout(dropout_ration)

        if self.current_dim==0:
            print(f"Dropout{self.current_layer_num}\t\t[None, {self.current_pixel}]")
            print("-"*50)
        else:
            print(f"Dropout{self.current_layer_num}\t\t[None, {self.current_dim}, {self.current_size}, {self.current_size}]")
            print("-"*50)

        self.current_layer_num += 1

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            # print(layer)
            # print(x.shape) 
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else :
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in self.trainLayers:
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx, (key, layer) in enumerate(self.layers.items()):
            """
            idx:何番目の層か
            """
            # print(key)
            if idx in self.trainLayers:
                if "BatchNorm" in key:
                    # print("BatchNorm update")
                    grads['W' + str(idx)] = layer.dgamma 
                    grads['b' + str(idx)] = layer.dbeta
                    self.params["MM"+str(idx)] = layer.moving_mean
                    self.params["MV"+str(idx)] = layer.moving_var
                else:
                    grads['W' + str(idx)] = layer.dW + self.weight_decay_lambda * self.params['W' + str(idx)]
                    grads['b' + str(idx)] = layer.db
        # print("grads:", grads.keys())

        return grads
    

class FineSimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 pool_param={'pool_size':2, 'pad':0, 'stride':2},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """
        input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : dict, 畳み込みの条件
        pool_param : dict, プーリングの条件
        hidden_size : int, 隠れ層のノード数
        output_size : int, 出力層のノード数
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        """
                
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        input_size = input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size = (conv_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        pool_output_pixel = filter_num * pool_output_size * pool_output_size # プーリング後のピクセル総数
        
        # 重みの初期化
        self.params = {}
        self.before_outputSize = output_size # 転移学習前の出力層のサイズ
        std = weight_init_std
        self.params['W1'] = std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size) # W1は畳み込みフィルターの重みになる
        self.params['b1'] = np.zeros(filter_num) #b1は畳み込みフィルターのバイアスになる
        self.params['W2'] = std *  np.random.randn(pool_output_pixel, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = std *  np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        gamma = np.ones(filter_num)
        beta = np.zeros(filter_num)
        self.layers['BachNorm1'] = BatchNormalization(gamma, beta)
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    # 転移学習用の層を追加
    def add(self, hidden_size,  output_size, weight_init_std=0.01):
        # self.key_fine = ["W4", "b4", "W5", "b5"]
        self.layers_fine = {
            "Affine3":["W4", "b4"],
            "Affine4":["W5", "b5"]
                            }

        std = weight_init_std
        # 重みの追加
        self.params['W4'] = std *  np.random.randn(self.before_outputSize , hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = std *  np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        # レイヤの追加
        self.layers['Affine3'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['ReLU3'] = ReLU()
        self.layers['Affine4'] = Affine(self.params['W5'], self.params['b5'])



    def predict(self, x):
        for layer in self.layers.values():
            # print(layer)
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t, fine=False):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        if fine==True:
            for layer, [w, b] in self.layers_fine.items():
                grads[w], grads[b] = self.layers[layer].dW, self.layers[layer].db

        return grads
