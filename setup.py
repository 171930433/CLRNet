import glob
import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 整个 parse_requirements 函数都可以被安全地删除，因为它没有被调用

def get_extensions():
    extensions = []

    # 找到所有需要编译的 C++/CUDA 源文件
    op_files = glob.glob('./clrnet/ops/csrc/*.c*')
    extension = CUDAExtension
    ext_name = 'clrnet.ops.nms_impl'

    # 定义扩展模块
    ext_ops = extension(
        name=ext_name,
        sources=op_files,
    )

    extensions.append(ext_ops)

    return extensions


setup(name='clrnet',
      version="1.0",
      keywords='computer vision & lane detection',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Intended Audience :: Developers',
          'Operating System :: OS Independent'
      ],
      packages=find_packages(),
      include_package_data=True,
      # 确认 install_requires 这一行不存在或被注释掉
      # install_requires=... , 
      
      # 以下是编译扩展的核心部分
      ext_modules=get_extensions(),
      cmdclass={'build_ext': BuildExtension},
      zip_safe=False)