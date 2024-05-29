import os
import shutil


# # def mymovefile(srcfile, dstpath):                       # 移动函数
# #     if not os.path.isfile(srcfile):
# #         print("%s not exist!" % (srcfile))
# #     else:
# #         fpath, fname = os.path.split(srcfile)             # 分离文件名和路径
# #         if not os.path.exists(dstpath):
# #             os.makedirs(dstpath)                       # 创建路径
# #         shutil.move(srcfile, dstpath + fname)          # 移动文件
# #         print("move %s -> %s"%(srcfile, dstpath + fname))
#
#
# def mymovefile(filename, file_type):                       # 移动函数
#     dataset, sample_num = filename.split("_")[0], filename.split("_")[1]
#     new_folder = os.path.join(dataset, sample_num, file_type)
#
#     if not os.path.isfile(filename):
#         print("{} not exist!".format(filename))
#     else:
#         # fpath, fname = os.path.split(srcfile)             # 分离文件名和路径
#         if not os.path.exists(new_folder):
#             os.makedirs(new_folder)                     # 创建路径
#         shutil.move(filename, os.path.join(new_folder, filename))          # 移动文件
#         # print("move %s -> %s" % (srcfile, dstpath + fname))
#
#
# def file_filter(files):
#     for filename in files:
#         if os.path.isfile(filename):
#             if filename.split(".")[-1] == "png" or filename.split(".")[-1] == "txt":
#                 mymovefile(filename, filename.split(".")[-1])
#     print("finish moving")
#
#
# files = os.listdir()
# file_filter(files)


def remove_files(folder):
    for root, folder, files in os.walk(folder):
        for file in files:
            if file.split(".")[-1] == "txt":
                print(file)
                pt_file_path = os.path.join(root, file)
                print(os.path.exists(pt_file_path))
                # print(root)
                # shutil.rmtree(pt_file_path)
                
                # os.remove(pt_file_path)


# remove_files("../../checkpoint/NIPS2023")
remove_files("./")