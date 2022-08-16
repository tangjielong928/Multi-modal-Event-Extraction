## -*- coding: utf-8 -*-
'''
计算评价指标P, R, F1
    F1 = (2 * P * R) / (P + R)
    P = 预测正确事件要素个数 / 全部预测事件要素数量
    R = 预测正确事件要素个数 / 全部正确标注要素数量
'''

class EvaluationMetric(object):
    def __init__(self, txt_pre='', txt_gt=''):
        self.pre_, self.count_pre_ = self.read_pre(txt_pre=txt_pre)
        self.gt_, self.count_gt_ = self.read_gt(txt_gt=txt_gt)

    def read_pre(self, txt_pre=''):
        with open(txt_pre, 'r') as f:
            lines = f.readlines()
        if lines:
            pre_dict = {}
            for line in lines:
                argus = line.strip().split('\t')
                # assert len(argus) == 6
                id = argus[0]
                if id in pre_dict.keys():
                    pre_dict[id].append(argus[1:6])
                else:
                    pre_dict[id] = [argus[1:6], ]
        return pre_dict, len(lines)

    def read_gt(self, txt_gt=''):
        with open(txt_gt, 'r') as f:
            lines = f.readlines()
        if lines:
            gt_dict = {}
            for line in lines:
                argus = line.strip().split('\t')
                # assert len(argus) == 6
                id = argus[0]
                if id in gt_dict.keys():
                    gt_dict[id].append(argus[1:6])
                else:
                    gt_dict[id] = [argus[1:6], ]
        return gt_dict, len(lines)

    def calculate_iou_1(self, bbox_pre='', bbox_gt=''):
        bbox_pre_pre = [float(it) for it in bbox_pre[1:-1].split(' ')]
        bbox_pre_gt = [float(it) for it in bbox_gt[1:-1].split(' ')]

        s_rec1 = (bbox_pre_pre[2] - bbox_pre_pre[0]) * (bbox_pre_pre[3] - bbox_pre_pre[1])  # 第一个bbox面积 = 长×宽
        s_rec2 = (bbox_pre_gt[2] - bbox_pre_gt[0]) * (bbox_pre_gt[3] - bbox_pre_gt[1])      # 第二个bbox面积 = 长×宽
        sum_s = s_rec1 + s_rec2                         # 总面积
        left = max(bbox_pre_pre[0], bbox_pre_gt[0])     # 并集左上角顶点横坐标
        right = min(bbox_pre_pre[2], bbox_pre_gt[2])    # 并集右下角顶点横坐标
        bottom = max(bbox_pre_pre[1], bbox_pre_gt[1])   # 并集左上角顶点纵坐标
        top = min(bbox_pre_pre[3], bbox_pre_gt[3])      # 并集右下角顶点纵坐标
        if left >= right or top <= bottom:             # 不存在并集的情况
            return 0
        else:
            inter = (right - left) * (top - bottom)  # 求并集面积
            iou = (inter / (sum_s - inter)) * 1.0  # 计算IOU
            return iou

    def calculate_iou_2(self, bbox_pre='', bbox_gt=''):
        bbox_pre_pre = [float(it) for it in bbox_pre[1:-1].split(' ')]
        bbox_pre_gt = [float(it) for it in bbox_gt[1:-1].split(' ')]
        x1, y1, x2, y2 = bbox_pre_pre  # box1的左上角坐标、右下角坐标
        x3, y3, x4, y4 = bbox_pre_gt  # box2的左上角坐标、右下角坐标

        # 计算交集的坐标
        x_inter1 = max(x1, x3)  # union的左上角x
        y_inter1 = max(y1, y3)  # union的左上角y
        x_inter2 = min(x2, x4)  # union的右下角x
        y_inter2 = min(y2, y4)  # union的右下角y

        # 计算交集部分面积，因为图像是像素点，所以计算图像的长度需要加一
        # 比如有两个像素点(0,0)、(1,0)，那么图像的长度是1-0+1=2，而不是1-0=1
        interArea = max(0, x_inter2 - x_inter1 + 1) * max(0, y_inter2 - y_inter1 + 1)

        # 分别计算两个box的面积
        area_box1 = (x2 - x1 + 1) * (y2 - y1 + 1)
        area_box2 = (x4 - x3 + 1) * (y4 - y3 + 1)

        # 计算IOU，交集比并集，并集面积=两个矩形框面积和-交集面积
        iou = interArea / (area_box1 + area_box2 - interArea)
        return iou

    def calculate_P_R_F1(self, pre={}, count_pre=0, gt={}, count_gt=0, flag_onlyText=True):
        P, R, F1 = 0., 0., 0.

        count_crrect = 0
        for key_pre, val_pre in pre.items():
            if key_pre not in gt.keys():
                continue
            else:
                val_gt = gt[key_pre]
                ## 开始遍历
                for i in range(len(val_pre)):
                    i_type, i_sta, i_end, i_argu, i_bbox = val_pre[i]
                    for j in range(len(val_gt)):
                        j_type, j_sta, j_end, j_argu, j_bbox = val_gt[j]
                        if i_type != j_type:
                            continue
                        else:
                            if i_sta != j_sta:
                                continue
                            else:
                                if i_end != j_end:
                                    continue
                                else:
                                    if flag_onlyText:
                                        count_crrect += 1
                                    else:
                                        if i_argu != j_argu:
                                            continue
                                        else:
                                            if i_bbox.startswith('-') and j_bbox.startswith('-'):
                                                count_crrect += 1
                                            elif i_bbox.startswith('-') and j_bbox.startswith('['):
                                                continue
                                            elif i_bbox.startswith('[') and j_bbox.startswith('-'):
                                                continue
                                            elif i_bbox.startswith('[') and j_bbox.startswith('['):
                                                iou = self.calculate_iou_2(i_bbox.strip(), j_bbox.strip())
                                                if iou <= 0.5:
                                                    continue
                                                else:
                                                    count_crrect += 1
        '''
        F1 = (2 * P * R) / (P + R)
        P = 预测正确事件要素个数 / 全部预测事件要素数量
        R = 预测正确事件要素个数 / 全部正确标注要素数量
        '''
        print("预测：{}\n标注：{}\n\n正确：{}\n".format(count_pre, count_gt, count_crrect))

        assert count_pre != 0
        assert count_gt != 0
        P = count_crrect / count_pre
        R = count_crrect / count_gt
        F1 = (2 * P * R) / (P + R)

        return P, R, F1

if __name__ == '__main__':
    em = EvaluationMetric(txt_pre='/data/zxwang/ccks2022/val/val_result/result_20220725_180219_clear.txt', txt_gt='../THUCNews/ccks2022/val_gt.txt')
    P, R, F1 = em.calculate_P_R_F1(em.pre_, em.count_pre_, em.gt_, em.count_gt_, flag_onlyText=False)
    print("P={:.5f}\tR={:.5f}\tF1={:.5f} \t".format(P, R, F1))