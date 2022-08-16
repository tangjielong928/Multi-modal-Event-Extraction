# -*- coding:utf-8 -*-
# @Time : 2022/7/14 10:25
# @Author: Jielong Tang
# @File : test.py
import paddle
from toolHandlerJLT import Handler

def testGPU():
    env_info = {}
    compiled_with_cuda = paddle.is_compiled_with_cuda()
    env_info['Paddle compiled with cuda'] = compiled_with_cuda
    print(paddle.get_device())

    if compiled_with_cuda:
        v = paddle.get_cudnn_version()
        v = str(v // 1000) + '.' + str(v % 1000 // 100)
        env_info['cudnn'] = v
        if 'gpu' in paddle.get_device():
            gpu_nums = paddle.distributed.ParallelEnv().nranks
        else:
            gpu_nums = 0
        env_info['GPUs used'] = gpu_nums

    env_info['PaddlePaddle'] = paddle.__version__

    for k, v in env_info.items():
        print('{}: {}'.format(k, v))

if __name__ == '__main__':
    print(Handler.sort_([]))
    # testGPU()
    # paddle.fluid.install_check.run_check()
    # paddle.fluid.is_compiled_with_cuda()
