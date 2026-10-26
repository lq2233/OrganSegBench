from totalsegmentator.python_api import totalsegmentator
import os
if __name__ == "__main__":
    print('begin')

    if not os.path.exists('./segmentation'):
        os.mkdir('./segmentation')
    a = os.system("TotalSegmentator -i " + "inp.nii.gz" + " -o " + "/segmentation --task total_mr")  # 使用a接收返回值

