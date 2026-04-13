#versao3-local.py - para computador local
#Imprime versao de TensorFlow/Keras.
#Tambem imprime o modelo do GPU e se TensorFlow consegue usa-lo.
#Imprime versao da Cuda, CUDNN, etc.
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import tensorflow.keras as keras
import sys, cv2
print("Versao python3:",sys.version)
print("Versao de tensorflow:",tf.__version__)
print("Versao de OpenCV:",cv2.__version__)
print()
gpu=tf.test.gpu_device_name()
if gpu=="":
 print("Computador sem GPU.")
else:
 print("Computador com GPU:",tf.test.gpu_device_name())
 from tensorflow.python.client import device_lib
 devices=device_lib.list_local_devices()
 print("Dispositivos:",[x.physical_device_desc for x in devices if x.physical_device_desc!=""])
 os.system("cat /usr/include/cudnn_version.h | grep '#define CUDNN_MAJOR' -A 2")
 os.system("nvcc --version | grep release")
 os.system('nvidia-smi')
print()
os.system('lsb_release -a | grep "Description"') #imprime qual é o sistema operacional
os.system('cat /proc/cpuinfo | grep -E "model name"') #especificações de CPU
os.system('cat /proc/meminfo | grep "Mem"') #especificações de RAM
print()
import torch;
print("Versao pytorch: ",torch.__version__);
print("GPU disponivel em pytorch: ",torch.cuda.is_available());
