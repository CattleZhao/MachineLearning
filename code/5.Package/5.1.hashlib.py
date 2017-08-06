# coding=utf-8

import hashlib

if __name__ == '__main__':
    md5 = hashlib.md5()
    md5.update('This is a sentence.'.encode("utf-8"))
    md5.update('This is a second sentence.'.encode("utf-8"))
    print(u'不出意外，这将是个乱码：', md5.digest())
    print(u'MD5：', md5.hexdigest())
    print(md5.digest_size, md5.block_size)

    print(hashlib.algorithms_available)