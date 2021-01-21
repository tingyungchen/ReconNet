# GPU 環境架設(tensorflow-gpu==1.14)

* 指令 `conda install tensorflow-gpu==1.14 `
* 下載CUDA 10.0套件
* 下載cuDNN for CUDA 10.0套件(需要官網註冊，我是直接用 google登入 )
* 把cudnn-10.0-windows10-x64-v7.1.zip解壓後，把bin,include,lib裡面的檔案(共三個)分別拷貝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\下的bin,include,lib目錄下即可。

# Quantization with Tensorflow

* 注意:v1 v2混用會失敗，然後windows不支援v1的quantization
* 要用quantization功能就要TF 大於2.3的版本
* 依照[TF官網的步驟](https://www.tensorflow.org/install/pip?hl=zh-tw#virtual-environment-install)依序安裝即可，如果遇到WINDOWS無法source venv的script時，直接開script檔案複製貼到terminal
* 注意以下所有動作都要用script啟動venv之後才可以用喔
*

# AttributeError: module 'tensorflow' has no attribute 'placeholder'

* 因為我的code有placeholder，同時又要用2.3以上版本的話做quantize，只能用`tf.compat.v1.placeholder()`

AttributeError: module 'tensorflow' has no attribute 'get_default_graph'

* `tf.compat.v1.get_default_graph()`

AttributeError: module 'tensorflow' has no attribute 'variable_scope

* `tf.compat.v1.variable_scope()`

AttributeError: module 'tensorflow' has no attribute 'get_variable'

* `tf.compat.v1.get_variable()`

同理大部分這個問題都可以如上的方式解決，以下條列一些需要特殊方式解決的

# AttributeError: module 'tensorflow_addons' has no attribute 'contrib'

* 這個比較麻煩，Because of TensorFlow 2.x module deprecations (for example, `tf.flags` and `tf.contrib`), some changes can not be worked around by switching to [`compat.v1`](https://www.tensorflow.org/api_docs/python/tf/compat/v1). Upgrading this code may require using an additional library (for example, [`absl.flags`](https://github.com/abseil/abseil-py)) or switching to a package in [tensorflow/addons](http://www.github.com/tensorflow/addons).
* 但後來發現更快的方法是直接把`tf.contrib.layers.xavier_initializer()`改成`tf.initializers.GlorotUniform()` (一樣的東西)
