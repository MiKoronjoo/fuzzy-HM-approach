import warnings

import numpy as np
import matplotlib.pyplot as plt
import xlrd


def hma(ls, rs):
    nans = np.where(np.isnan(ls) + np.isnan(rs))
    ls = np.delete(ls, nans)
    rs = np.delete(rs, nans)

    del_indexes = [i for i in range(len(ls) - 1, -1, -1)
                   if ls[i] < 0 or ls[i] > 10 or rs[i] < 0 or rs[i] > 10 or rs[i] <= ls[i] or rs[i] - ls[i] >= 10]
    ls = np.delete(ls, del_indexes)
    rs = np.delete(rs, del_indexes)

    nums = len(ls)
    int_leng = rs - ls
    sorted_left = np.sort(ls)
    sorted_right = np.sort(rs)
    leng = np.sort(int_leng)
    NN1 = int(nums * 0.25)
    NN2 = int(nums * 0.75)

    ql25 = sorted_left[NN1 - 1] * (1 - (0.25 * nums % 1)) + sorted_left[NN1] * (0.25 * nums % 1)
    ql75 = sorted_left[NN2 - 1] * (1 - (0.75 * nums % 1)) + sorted_left[NN2] * (0.75 * nums % 1)
    liqr = ql75 - ql25

    qr25 = sorted_right[NN1 - 1] * (1 - (0.25 * nums % 1)) + sorted_right[NN1] * (0.25 * nums % 1)
    qr75 = sorted_right[NN2 - 1] * (1 - (0.75 * nums % 1)) + sorted_right[NN2] * (0.75 * nums % 1)
    riqr = qr75 - qr25

    del_indexes = [i for i in range(nums - 1, -1, -1) if
                   ls[i] < ql25 - 1.5 * liqr or ls[i] > ql75 + 1.5 * liqr or
                   rs[i] < qr25 - 1.5 * riqr or rs[i] > qr75 + 1.5 * riqr]
    ls = np.delete(ls, del_indexes)
    rs = np.delete(rs, del_indexes)
    int_leng = np.delete(int_leng, del_indexes)

    nums = len(ls)
    NN1 = int(nums * 0.25)
    NN2 = int(nums * 0.75)
    q_leng25 = leng[NN1 - 1] * (1 - (0.25 * nums % 1)) + leng[NN1] * (0.25 * nums % 1)
    q_leng75 = leng[NN2 - 1] * (1 - (0.75 * nums % 1)) + leng[NN2] * (0.75 * nums % 1)
    lengIQR = q_leng75 - q_leng25

    del_indexes = [i for i in range(len(ls) - 1, -1, -1)
                   if int_leng[i] < q_leng25 - 1.5 * lengIQR or int_leng[i] > q_leng75 + 1.5 * lengIQR]
    ls = np.delete(ls, del_indexes)
    rs = np.delete(rs, del_indexes)
    int_leng = np.delete(int_leng, del_indexes)

    mean_ls = ls.mean()
    std_ls = ls.std()
    mean_rs = rs.mean()
    std_rs = rs.std()
    k = K[min(len(ls), 25) - 1]
    del_indexes = [i for i in range(len(ls) - 1, -1, -1) if
                   ls[i] < mean_ls - k * std_ls or ls[i] > mean_ls + k * std_ls or
                   rs[i] < mean_rs - k * std_rs or rs[i] > mean_rs + k * std_rs]
    ls = np.delete(ls, del_indexes)
    rs = np.delete(rs, del_indexes)
    int_leng = np.delete(int_leng, del_indexes)

    mean_leng = int_leng.mean()
    std_leng = int_leng.std()
    k = min([k, mean_leng / std_leng, (10 - mean_leng) / std_leng])
    del_indexes = [i for i in range(len(ls) - 1, -1, -1)
                   if int_leng[i] < mean_leng - k * std_leng or int_leng[i] > mean_leng + k * std_leng]
    ls = np.delete(ls, del_indexes)
    rs = np.delete(rs, del_indexes)

    mean_ls = np.mean(ls)
    std_ls = np.std(ls)
    mean_rs = np.mean(rs)
    std_rs = np.std(rs)
    if std_ls == std_rs:
        barrier = (mean_ls + mean_rs) / 2
    elif std_ls == 0:
        barrier = mean_ls + 0.01
    elif std_rs == 0:
        barrier = mean_rs - 0.01
    else:
        barrier1 = (mean_rs * std_ls ** 2 - mean_ls * std_rs ** 2 + std_ls * std_rs * np.sqrt(
            (mean_ls - mean_rs) ** 2 + 2 * (std_ls ** 2 - std_rs ** 2) * np.log(std_ls / std_rs))) / (
                           std_ls ** 2 - std_rs ** 2)
        barrier2 = (mean_rs * std_ls ** 2 - mean_ls * std_rs ** 2 - std_ls * std_rs * np.sqrt(
            (mean_ls - mean_rs) ** 2 + 2 * (std_ls ** 2 - std_rs ** 2) * np.log(std_ls / std_rs))) / (
                           std_ls ** 2 - std_rs ** 2)
        barrier = barrier1 if mean_ls <= barrier1 <= mean_rs else barrier2

    del_indexes = [i for i in range(len(ls) - 1, -1, -1) if
                   ls[i] >= barrier or rs[i] <= barrier or
                   ls[i] < 2 * mean_ls - barrier or rs[i] > 2 * mean_rs - barrier]
    ls = np.delete(ls, del_indexes)
    rs = np.delete(rs, del_indexes)

    mean_ls = np.mean(ls)
    std_ls = np.std(ls)
    mean_rs = np.mean(rs)
    std_rs = np.std(rs)
    k = K[min(len(ls), 25) - 1]

    if mean_ls - k * std_ls <= 0:
        # LEFT SHOULDER
        switch_point = min(rs)
        if np.sum(rs - switch_point) == 0:
            return

        c = switch_point
        subset_right_length = rs - c

        subset_right_length = subset_right_length[subset_right_length > 0]

        d = min(c + 3 * np.sqrt(2) * subset_right_length.std(), 10)
        i = min(6 * (c + subset_right_length.mean()) - 4 * c - d, 10)

        bl = max(min(d, i), c)
        br = max(d, i)

        umf = [0, 0, switch_point, br]
        lmf = [0, 0, switch_point, bl, 1]

    elif mean_rs + k * std_rs >= 10:
        # RIGHT SHOULDER
        switch_point = max(ls)
        if np.sum(ls - switch_point) == 0:
            return
        c = switch_point
        subset_right_length = c - ls
        subset_right_length = subset_right_length[subset_right_length > 0]

        a = max(0, c - 3 * np.sqrt(2) * subset_right_length.std())
        e = max(6 * (c - subset_right_length.mean()) - 4 * c - a, 0)
        al = min(a, e)
        ar = max(e, a)
        umf = [al, switch_point, 10, 10]
        lmf = [ar, switch_point, 10, 10, 1]

    else:
        # INTERIOR
        overlap_left = max(ls)
        overlap_right = min(rs)

        c = overlap_left
        subset_right_length = c - ls
        subset_right_length = subset_right_length[subset_right_length > 0]

        a = max(0, c - 3 * np.sqrt(2) * subset_right_length.std())
        e = max(0, 6 * (c - subset_right_length.mean()) - 4 * c - a)

        al = min(a, e)
        ar = min(max(e, a), c)

        c = overlap_right
        subset_right_length = rs - c
        subset_right_length = subset_right_length[subset_right_length > 0]

        d = min(c + 3 * np.sqrt(2) * subset_right_length.std(), 10)
        i = min(6 * (c + subset_right_length.mean()) - 4 * c - d, 10)

        bl = max(min(d, i), c)
        br = max(d, i)

        umf = [al, overlap_left, overlap_right, br]
        lmf = [ar, overlap_left, overlap_right, bl, 1]

    return umf, lmf


def read_dataset(path: str, words_count: int, records: int):
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_index(0)
    data = []
    words = []
    for j in range(0, words_count * 2, 2):
        nums = []
        words.append(worksheet.cell_value(0, j))
        for i in range(1, records + 1):
            num_l = worksheet.cell_value(i, j)
            num_r = worksheet.cell_value(i, j + 1)
            nums.append((num_l, num_r))
        data.append([n for rng in sorted(nums) for n in rng])
    return np.array(data), words


def plot(umf, lmf, word: str):
    x = np.linspace(0, 10, 1000)
    umf_values = np.interp(x, [umf[0], umf[1], umf[2], umf[3]], [0, 1, 1, 0])
    plt.plot(x, umf_values, label='UMF', color='darkorange')
    lmf_values = np.interp(x, [lmf[0], lmf[1], lmf[2], lmf[3]], [0, 1, 1, 0])
    plt.plot(x, lmf_values, label='LMF', color='blueviolet')
    plt.fill_between(x, umf_values, lmf_values, where=(umf_values > lmf_values),
                     interpolate=True, color='gray', alpha=0.8)
    plt.title(word)
    plt.xlabel('Value')
    plt.ylabel('Membership')
    plt.legend()
    plt.show()


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    print('1. DATACOPY-28')
    print('2. WEBdatacopy-174')
    choice = input('Choose the dataset [1/2]: ')
    if choice == '1':
        path = DATASET1_PATH
        records = RECORDS1
    elif choice == '2':
        path = DATASET2_PATH
        records = RECORDS2
    else:
        print('Invalid input. DATACOPY-28 will load by default')
        path = DATASET1_PATH
        records = RECORDS1
    data, words = read_dataset(path, WORDS_COUNT, records)
    for i, word in enumerate(words):
        print(f'word {i + 1}: {word}')
        ls = data[i][::2]
        rs = data[i][1::2]
        result = hma(ls, rs)
        if result is None:
            continue
        umf, lmf = result
        plot(umf, lmf, word)


K = [32.019, 32.019, 8.380, 5.369, 4.275, 3.712, 3.369, 3.136, 2.967, 2.839,
     2.737, 2.655, 2.587, 2.529, 2.480, 2.437, 2.400, 2.366, 2.337, 2.310,
     2.310, 2.310, 2.310, 2.310, 2.208]
DATASET1_PATH = 'datasets/DATACOPY-28.XLS'
RECORDS1 = 28
DATASET2_PATH = 'datasets/WEBdatacopy-174.xls'
RECORDS2 = 174
WORDS_COUNT = 32

if __name__ == '__main__':
    main()
